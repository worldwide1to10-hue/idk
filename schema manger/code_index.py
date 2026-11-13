"""Incremental Tree-sitter symbol/call index with task-scope isolation."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tools.search import (
    MAX_SEARCH_FILE_BYTES,
    MAX_SEARCH_FILES,
    MAX_SEARCH_TOTAL_BYTES,
    _is_excluded_path,
    _iter_project_files,
    _project_root,
    _resolve_search_path,
)


INDEX_SCHEMA_VERSION = "2"
MAX_INDEX_FILES = 5_000
MAX_INDEX_TOTAL_BYTES = 128 * 1024 * 1024
CODE_SUFFIXES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
}

SYMBOL_NODES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "typescript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
    },
    "tsx": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
    },
    "javascript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_spec": "type",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
    },
}
CALL_NODES = frozenset({"call", "call_expression"})


@dataclass(frozen=True)
class IndexedFile:
    path: str
    relative_path: str
    language: str
    digest: str
    content: bytes
    root_node: Any


@dataclass(frozen=True)
class IndexSummary:
    database_path: str
    files_indexed: int
    files_reused: int
    files_removed: int
    symbols: int
    call_candidates: int
    import_edges: int
    parse_warnings: int
    skipped_files: int
    scope_limited: bool


def _index_path(root: str, scope_identity: str) -> str:
    digest = hashlib.sha256()
    digest.update(os.path.realpath(root).encode("utf-8"))
    digest.update(b"\0")
    digest.update(scope_identity.encode("utf-8"))
    key = digest.hexdigest()
    state_root = os.environ.get(
        "AGENT_CODE_INDEX_DIR",
        os.path.join(os.path.expanduser("~"), ".agent", "code-index"),
    )
    directory = os.path.realpath(os.path.expanduser(state_root))
    os.makedirs(directory, mode=0o700, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    return os.path.join(directory, f"{key}.sqlite3")


def _connect(database_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(
        database_path,
        timeout=15,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            digest TEXT NOT NULL,
            language TEXT NOT NULL,
            parse_error INTEGER NOT NULL,
            indexed_at REAL NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            signature TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS symbols_name_idx ON symbols(name);
        CREATE INDEX IF NOT EXISTS symbols_qualified_idx ON symbols(qualified_name);
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY,
            caller_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
            target_name TEXT NOT NULL,
            line INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS calls_target_idx ON calls(target_name);
        CREATE INDEX IF NOT EXISTS calls_caller_idx ON calls(caller_id);
        CREATE TABLE IF NOT EXISTS imports (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
            imported_module TEXT NOT NULL,
            line INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS imports_module_idx ON imports(imported_module);
    """)
    previous_version = connection.execute(
        "SELECT value FROM metadata WHERE key='schema_version'",
    ).fetchone()
    if previous_version is not None and previous_version["value"] != INDEX_SCHEMA_VERSION:
        connection.execute("DELETE FROM files")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (INDEX_SCHEMA_VERSION,),
    )
    connection.commit()
    if database_path != ":memory:":
        try:
            os.chmod(database_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    return connection


def _language_for_path(path: str) -> str | None:
    return CODE_SUFFIXES.get(Path(path).suffix.lower())


def _safe_node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_name(node: Any, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        declarator = node.child_by_field_name("declarator")
        if declarator is not None:
            name_node = declarator.child_by_field_name("name")
    if name_node is None:
        return None
    value = _safe_node_text(name_node, source).strip()
    return value if value and value.isidentifier() else None


def _short_signature(node: Any, source: bytes) -> str:
    body = node.child_by_field_name("body")
    end_byte = body.start_byte if body is not None else node.end_byte
    raw = source[node.start_byte:end_byte].decode("utf-8", errors="replace")
    return " ".join(raw.strip().split())[:600]


def _call_target(node: Any, source: bytes) -> str | None:
    target = node.child_by_field_name("function")
    if target is None:
        target = node.child_by_field_name("method")
    if target is None and node.named_children:
        target = node.named_children[0]
    if target is None:
        return None
    text = _safe_node_text(target, source).strip()
    leaf = text.replace("::", ".").split(".")[-1]
    return leaf if leaf.isidentifier() else None


def _extract_symbols(root_node: Any, source: bytes, language: str) -> list[dict[str, Any]]:
    node_kinds = SYMBOL_NODES.get(language, {})
    symbols: list[dict[str, Any]] = []
    parent_names: list[str] = []

    def visit(node: Any) -> None:
        kind = node_kinds.get(node.type)
        name = _node_name(node, source) if kind else None
        if kind and name:
            qualified = ".".join((*parent_names, name))
            record = {
                "node_id": node.id,
                "name": name,
                "qualified_name": qualified,
                "kind": kind,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
                "signature": _short_signature(node, source),
            }
            symbols.append(record)
            parent_names.append(name)
            for child in node.children:
                visit(child)
            parent_names.pop()
            return
        for child in node.children:
            visit(child)

    visit(root_node)
    return symbols


def _extract_calls(root_node: Any, source: bytes, symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    symbol_by_node_id = {symbol["node_id"]: symbol for symbol in symbols}
    calls: list[dict[str, Any]] = []

    def visit(node: Any, owner: dict[str, Any] | None) -> None:
        current_owner = symbol_by_node_id.get(node.id, owner)
        if node.type in CALL_NODES and current_owner is not None:
            target_name = _call_target(node, source)
            if target_name:
                calls.append({
                    "caller_qualified_name": current_owner["qualified_name"],
                    "target_name": target_name,
                    "line": node.start_point.row + 1,
                })
        for child in node.children:
            visit(child, current_owner)

    visit(root_node, None)
    return calls


def _extract_imports(root_node: Any, source: bytes, language: str) -> list[dict[str, Any]]:
    imports: list[dict[str, Any]] = []
    import_nodes = {
        "python": {"import_statement", "import_from_statement"},
        "typescript": {"import_statement"},
        "tsx": {"import_statement"},
        "javascript": {"import_statement"},
        "go": {"import_spec"},
        "rust": {"use_declaration"},
    }.get(language, set())
    for node in _walk_nodes(root_node):
        if node.type not in import_nodes:
            continue
        if language == "python":
            import_value = (
                node.child_by_field_name("module_name")
                if node.type == "import_from_statement"
                else node.child_by_field_name("name")
            )
            value = _safe_node_text(import_value, source) if import_value else ""
        elif language in ("typescript", "tsx", "javascript"):
            import_value = node.child_by_field_name("source")
            value = _safe_node_text(import_value, source).strip("\"'` ") if import_value else ""
        elif language == "go":
            import_value = node.child_by_field_name("path")
            value = _safe_node_text(import_value, source).strip("\"' ") if import_value else ""
        else:
            import_value = node.child_by_field_name("argument")
            value = _safe_node_text(import_value, source) if import_value else ""
        value = value.strip()
        if value:
            imports.append({"module": value, "line": node.start_point.row + 1})
    return imports


def _walk_nodes(root_node: Any) -> Iterator[Any]:
    stack = [root_node]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _iter_source_files(context: Any, root: str, base_path: str) -> Iterator[tuple[str, str, str]]:
    count = 0
    total_bytes = 0
    for path, display_path in _iter_project_files(context, root, base_path):
        if count >= min(MAX_INDEX_FILES, MAX_SEARCH_FILES):
            break
        if _is_excluded_path(root, display_path):
            continue
        language = _language_for_path(display_path)
        if language is None:
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > MAX_SEARCH_FILE_BYTES or total_bytes + size > MAX_INDEX_TOTAL_BYTES:
            continue
        count += 1
        total_bytes += size
        yield path, display_path, language


def build_code_index(context: Any, requested_path: str = ".") -> tuple[sqlite3.Connection, IndexSummary]:
    root, base_path = _resolve_search_path(context, requested_path)
    scope = getattr(context, "file_scope", None)
    scope_limited = scope is not None
    files = list(_iter_source_files(context, root, base_path))
    if scope_limited:
        scoped_paths = sorted(
            os.path.relpath(path, root)
            for path in scope.read_paths | scope.write_paths
        )
        scope_identity = "manifest:" + "\0".join(scoped_paths)
    else:
        scope_identity = "path:" + os.path.relpath(base_path, root)
    database_path = ":memory:" if scope_limited else _index_path(root, scope_identity)
    connection = _connect(database_path)

    previous_paths = {
        row["path"] for row in connection.execute("SELECT path FROM files")
    }
    current_paths = {os.path.realpath(display) for _path, display, _lang in files}
    removed = len(previous_paths - current_paths)
    for stale_path in previous_paths - current_paths:
        connection.execute("DELETE FROM files WHERE path=?", (stale_path,))

    indexed = 0
    reused = 0
    parse_warnings = 0
    skipped = 0
    for search_path, display_path, language in files:
        canonical = os.path.realpath(display_path)
        try:
            with open(search_path, "rb") as source_file:
                source = source_file.read(MAX_SEARCH_FILE_BYTES + 1)
        except OSError:
            skipped += 1
            continue
        if len(source) > MAX_SEARCH_FILE_BYTES or b"\x00" in source:
            skipped += 1
            continue
        digest = hashlib.sha256(source).hexdigest()
        existing = connection.execute(
            "SELECT digest FROM files WHERE path=?", (canonical,),
        ).fetchone()
        if existing and existing["digest"] == digest:
            reused += 1
            continue

        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(language)
            tree = parser.parse(source)
        except Exception:
            skipped += 1
            continue
        parse_error = int(bool(tree.root_node.has_error))
        parse_warnings += parse_error
        symbols = _extract_symbols(tree.root_node, source, language)
        calls = _extract_calls(tree.root_node, source, symbols)
        imports = _extract_imports(tree.root_node, source, language)

        connection.execute("DELETE FROM files WHERE path=?", (canonical,))
        connection.execute(
            "INSERT INTO files(path,digest,language,parse_error) VALUES(?,?,?,?)",
            (canonical, digest, language, parse_error),
        )
        symbol_ids: dict[str, int] = {}
        for symbol in symbols:
            cursor = connection.execute(
                """INSERT INTO symbols(
                    file_path,name,qualified_name,kind,start_line,end_line,signature
                ) VALUES(?,?,?,?,?,?,?)""",
                (
                    canonical, symbol["name"], symbol["qualified_name"],
                    symbol["kind"], symbol["start_line"], symbol["end_line"],
                    symbol["signature"],
                ),
            )
            symbol_ids[symbol["qualified_name"]] = cursor.lastrowid
        for call in calls:
            caller_id = symbol_ids.get(call["caller_qualified_name"])
            if caller_id is not None:
                connection.execute(
                    "INSERT INTO calls(caller_id,target_name,line) VALUES(?,?,?)",
                    (caller_id, call["target_name"], call["line"]),
                )
        for imported in imports:
            connection.execute(
                "INSERT INTO imports(file_path,imported_module,line) VALUES(?,?,?)",
                (canonical, imported["module"], imported["line"]),
            )
        indexed += 1

    connection.commit()
    counts = connection.execute(
        "SELECT (SELECT COUNT(*) FROM symbols) AS symbols, "
        "(SELECT COUNT(*) FROM calls) AS calls, "
        "(SELECT COUNT(*) FROM imports) AS imports, "
        "(SELECT COUNT(*) FROM files WHERE parse_error=1) AS parse_warnings"
    ).fetchone()
    return connection, IndexSummary(
        database_path=database_path,
        files_indexed=indexed,
        files_reused=reused,
        files_removed=removed,
        symbols=counts["symbols"],
        call_candidates=counts["calls"],
        import_edges=counts["imports"],
        parse_warnings=counts["parse_warnings"],
        skipped_files=skipped,
        scope_limited=scope_limited,
    )


def find_callers(connection: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    leaf = symbol.rsplit(".", 1)[-1]
    rows = connection.execute(
        """SELECT caller.file_path, caller.qualified_name, caller.start_line,
                  calls.line, calls.target_name
           FROM calls
           JOIN symbols AS caller ON caller.id = calls.caller_id
           WHERE calls.target_name=?
           ORDER BY caller.file_path, calls.line
           LIMIT 200""",
        (leaf,),
    )
    return [dict(row) for row in rows]


def find_callees(connection: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """SELECT caller.file_path, caller.qualified_name, calls.target_name,
                  calls.line, target.file_path AS candidate_file,
                  target.qualified_name AS candidate_symbol
           FROM symbols AS caller
           JOIN calls ON calls.caller_id = caller.id
           LEFT JOIN symbols AS target ON target.name = calls.target_name
           WHERE caller.qualified_name=? OR caller.name=?
           ORDER BY caller.file_path, calls.line, target.file_path
           LIMIT 200""",
        (symbol, symbol.rsplit(".", 1)[-1]),
    )
    return [dict(row) for row in rows]


def find_imports(connection: sqlite3.Connection, file_path: str) -> list[dict[str, Any]]:
    canonical = os.path.realpath(file_path)
    rows = connection.execute(
        """SELECT imported_module,line FROM imports
           WHERE file_path=? ORDER BY line, imported_module LIMIT 500""",
        (canonical,),
    )
    return [dict(row) for row in rows]


def ranked_symbols(connection: sqlite3.Connection, limit: int = 500) -> list[dict[str, Any]]:
    rows = [dict(row) for row in connection.execute(
        "SELECT id,file_path,name,qualified_name,kind,start_line,end_line,signature FROM symbols"
    )]
    if not rows:
        return []
    by_name: dict[str, list[int]] = {}
    positions = {row["id"]: index for index, row in enumerate(rows)}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row["id"])

    adjacency: dict[int, set[int]] = {row["id"]: set() for row in rows}
    for call in connection.execute("SELECT caller_id,target_name FROM calls"):
        candidates = by_name.get(call["target_name"], [])
        if len(candidates) > 20:
            continue
        for target_id in candidates:
            adjacency[call["caller_id"]].add(target_id)

    files = {
        row["path"]: row["language"]
        for row in connection.execute("SELECT path,language FROM files")
    }
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_file.setdefault(row["file_path"], []).append(row)
    for record in connection.execute("SELECT file_path,imported_module FROM imports"):
        source_path = record["file_path"]
        imported = record["imported_module"]
        candidates: set[str] = set()
        source_language = files.get(source_path)
        source_dir = os.path.dirname(source_path)
        if imported.startswith("."):
            relative = os.path.normpath(os.path.join(source_dir, imported))
            for suffix in CODE_SUFFIXES:
                candidates.add(relative + suffix)
                candidates.add(os.path.join(relative, "index" + suffix))
                candidates.add(os.path.join(relative, "mod" + suffix))
        elif source_language == "python":
            relative = imported.replace(".", os.sep)
            for suffix in (".py", ".pyi"):
                candidates.add(os.path.join(os.path.dirname(source_path), relative + suffix))
                candidates.add(os.path.join(os.path.dirname(os.path.dirname(source_path)), relative + suffix))
                candidates.add(os.path.join(os.path.dirname(source_path), relative, "__init__" + suffix))
        else:
            normalized = imported.replace("::", os.sep).replace(".", os.sep)
            for suffix in CODE_SUFFIXES:
                candidates.add(os.path.join(source_dir, normalized + suffix))
                candidates.add(os.path.join(source_dir, normalized, "index" + suffix))
        target_path = next((path for path in candidates if path in by_file), None)
        if target_path is not None:
            for source_symbol in by_file.get(source_path, []):
                for target_symbol in by_file.get(target_path, []):
                    adjacency[source_symbol["id"]].add(target_symbol["id"])

    scores = {row["id"]: 1.0 / len(rows) for row in rows}
    damping = 0.85
    for _iteration in range(16):
        updated = {row["id"]: (1.0 - damping) / len(rows) for row in rows}
        for source_id, targets in adjacency.items():
            if not targets:
                share = damping * scores[source_id] / len(rows)
                for target_id in updated:
                    updated[target_id] += share
                continue
            share = damping * scores[source_id] / len(targets)
            for target_id in targets:
                updated[target_id] += share
        scores = updated

    for row in rows:
        row["rank"] = scores[row["id"]]
        row["candidate_edges"] = len(adjacency[row["id"]])
    rows.sort(key=lambda row: (-row["rank"], row["file_path"], row["start_line"]))
    return rows[:limit]


def close_code_index(connection: sqlite3.Connection) -> None:
    connection.close()


__all__ = [
    "IndexSummary", "build_code_index", "find_callers", "find_callees",
    "find_imports", "close_code_index", "CODE_SUFFIXES",
]
