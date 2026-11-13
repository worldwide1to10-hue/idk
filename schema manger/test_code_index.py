"""Offline tests for incremental, scoped Tree-sitter indexing."""
import os
import sqlite3
import stat
import tempfile

from code_index import (
    build_code_index, close_code_index, find_callers, find_callees, find_imports,
)
from state import AppState, FileStateCache, ToolPermissionContext
from task_scheduler import FileLeaseRegistry, TaskFileScope
from tool import ToolUseContext


def make_context(root: str, scope: TaskFileScope | None = None) -> ToolUseContext:
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    context = ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda _update: None,
        read_file_state=state.read_file_cache,
    )
    context.project_root = root
    context.file_scope = scope
    return context


def main() -> None:
    previous_index_dir = os.environ.get("AGENT_CODE_INDEX_DIR")
    with tempfile.TemporaryDirectory() as root:
        index_dir = os.path.join(root, "private-index")
        os.environ["AGENT_CODE_INDEX_DIR"] = index_dir
        py_dir = os.path.join(root, "src")
        os.makedirs(py_dir)
        samples = {
            "src/app.py": (
                "from src.helpers import helper\n\n"
                "def helper(value):\n    return value\n\n"
                "def main():\n    return helper(1)\n"
            ),
            "src/web.ts": (
                "export function helper(value: number) { return value; }\n"
                "export function main() { return helper(1); }\n"
            ),
            "src/lib.go": (
                "package sample\nfunc helper(value int) int { return value }\n"
                "func main() { helper(1) }\n"
            ),
            "src/lib.rs": (
                "fn helper(value: i32) -> i32 { value }\n"
                "fn main() { helper(1); }\n"
            ),
        }
        paths = {}
        for relative, content in samples.items():
            path = os.path.join(root, relative)
            with open(path, "w", encoding="utf-8") as target:
                target.write(content)
            paths[relative] = path

        context = make_context(root)
        connection, first = build_code_index(context)
        assert first.files_indexed == 4, first
        assert first.symbols >= 8, first
        assert first.call_candidates >= 4, first
        assert first.import_edges >= 1, first
        assert first.parse_warnings == 0, first
        assert stat.S_IMODE(os.stat(first.database_path).st_mode) == 0o600
        caller_results = find_callers(connection, "helper")
        assert len(caller_results) >= 4, caller_results
        callee_results = find_callees(connection, "main")
        assert any(row["target_name"] == "helper" for row in callee_results)
        import_results = find_imports(connection, os.path.realpath(paths["src/app.py"]))
        assert any(row["imported_module"] == "src.helpers" for row in import_results)
        database_path = first.database_path
        close_code_index(connection)
        print("✅ Tree-Sitter indexes Python, TypeScript, Go, and Rust")

        connection, second = build_code_index(context)
        assert second.files_indexed == 0, second
        assert second.files_reused == 4, second
        assert second.database_path == database_path
        close_code_index(connection)
        print("✅ Content hashes reuse unchanged files")

        legacy = sqlite3.connect(database_path)
        legacy.execute(
            "UPDATE metadata SET value='1' WHERE key='schema_version'",
        )
        legacy.commit()
        legacy.close()
        connection, migrated = build_code_index(context)
        assert migrated.files_indexed == 4, migrated
        assert migrated.files_reused == 0, migrated
        close_code_index(connection)
        print("✅ Schema upgrades rebuild stale index rows")

        with open(paths["src/app.py"], "a", encoding="utf-8") as target:
            target.write("\ndef extra():\n    return helper(2)\n")
        new_path = os.path.join(root, "src", "new.ts")
        with open(new_path, "w", encoding="utf-8") as target:
            target.write("export function fresh() { return 1; }\n")
        connection, third = build_code_index(context)
        assert third.files_indexed == 2, third
        assert third.files_reused == 3, third
        assert third.database_path == database_path
        close_code_index(connection)
        print("✅ Changed and new files incrementally refresh the same index")

        broken_path = os.path.join(root, "src", "broken.py")
        with open(broken_path, "w", encoding="utf-8") as target:
            target.write("def incomplete(:\n    return 3\n")
        connection, broken = build_code_index(context)
        assert broken.parse_warnings >= 1, broken
        assert broken.files_indexed == 1, broken
        close_code_index(connection)
        print("✅ Syntax errors are reported while partial trees are indexed")

        scoped_path = os.path.realpath(paths["src/app.py"])
        scope = TaskFileScope(
            task_id="reader",
            project_root=os.path.realpath(root),
            read_paths=frozenset({scoped_path}),
            write_paths=frozenset(),
            leases=FileLeaseRegistry(root),
        )
        scoped_context = make_context(root, scope)
        scoped_connection, scoped = build_code_index(scoped_context)
        assert scoped.scope_limited
        assert scoped.database_path == ":memory:"
        indexed_paths = {
            row["path"] for row in scoped_connection.execute("SELECT path FROM files")
        }
        assert indexed_paths == {scoped_path}, indexed_paths
        close_code_index(scoped_connection)
        print("✅ Task manifest uses an isolated in-memory index")

    if previous_index_dir is None:
        os.environ.pop("AGENT_CODE_INDEX_DIR", None)
    else:
        os.environ["AGENT_CODE_INDEX_DIR"] = previous_index_dir
    print("\n🎉 All code-index tests passed")


main()
