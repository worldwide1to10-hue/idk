"""Python AST context packages and a bounded project symbol map."""
from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import tempfile
import stat
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field, field_validator

from permission_types import PermissionAllowDecision
from tool import ToolResult, ToolUseContext, build_tool
from code_index import (
    build_code_index,
    close_code_index,
    find_callers,
    find_callees,
    find_imports,
    ranked_symbols,
)
from lsp_client import JsonRpcStdioClient
from tools.search import (
    MAX_SEARCH_FILE_BYTES,
    MAX_SEARCH_FILES,
    MAX_SEARCH_TOTAL_BYTES,
    _iter_project_files,
    _is_excluded_path,
    _project_root,
    _relative_display_path,
    _resolve_search_path,
)


MAX_EDIT_CONTEXT_TOKENS = 4_000
MAX_EDIT_CONTEXT_CALLERS = 12
MAX_EDIT_CONTEXT_HELPER_LINES = 25
MAX_REPO_MAP_FILES = 2_000
MAX_AST_GREP_FILES = 100
MAX_AST_GREP_OUTPUT_BYTES = 2 * 1024 * 1024
AST_GREP_TIMEOUT_SECONDS = 20
AST_GREP_LANGUAGES = {
    "python": "python",
    "typescript": "ts",
    "tsx": "tsx",
    "javascript": "js",
    "go": "go",
    "rust": "rust",
}


class EditContextInput(BaseModel):
    file_path: str = Field(min_length=1, max_length=4096)
    symbol: str = Field(min_length=1, max_length=512)
    token_budget: int = Field(default=3_000, ge=256, le=MAX_EDIT_CONTEXT_TOKENS)
    include_callers: bool = True

    @field_validator("symbol")
    @classmethod
    def _trim_symbol(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("symbol must not be empty")
        return value


class RepoMapInput(BaseModel):
    path: str = Field(default=".", max_length=4096)
    token_budget: int = Field(default=1_024, ge=128, le=4_000)


class FindReferencesInput(BaseModel):
    file_path: str = Field(min_length=1, max_length=4096)
    symbol: str = Field(min_length=1, max_length=512)

    @field_validator("symbol")
    @classmethod
    def _trim_reference_symbol(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("symbol must not be empty")
        return value


class IndexCodeInput(BaseModel):
    path: str = Field(default=".", max_length=4096)


class SymbolQueryInput(BaseModel):
    path: str = Field(default=".", max_length=4096)
    symbol: str = Field(min_length=1, max_length=512)

    @field_validator("symbol")
    @classmethod
    def _trim_query_symbol(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("symbol must not be empty")
        return value


class FileQueryInput(BaseModel):
    file_path: str = Field(min_length=1, max_length=4096)


FindImportsInput = FileQueryInput


class AstGrepInput(BaseModel):
    pattern: str = Field(min_length=1, max_length=512)
    language: Literal["python", "typescript", "tsx", "javascript", "go", "rust"]
    path: str = Field(default=".", max_length=4096)
    max_results: int = Field(default=30, ge=1, le=100)

    @field_validator("pattern")
    @classmethod
    def _trim_ast_pattern(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("AST pattern must not be empty")
        return value


class PyrightNavigationInput(BaseModel):
    file_path: str = Field(min_length=1, max_length=4096)
    symbol: str = Field(min_length=1, max_length=512)

    @field_validator("symbol")
    @classmethod
    def _trim_pyright_symbol(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("symbol must not be empty")
        return value


class _ParsedFile:
    def __init__(self, path: str, display_path: str, source: str, tree: ast.Module):
        self.path = path
        self.display_path = display_path
        self.source = source
        self.lines = source.splitlines()
        self.tree = tree


def _read_python_file(path: str, display_path: str) -> _ParsedFile | None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_SEARCH_FILE_BYTES:
            return None
        with open(path, "r", encoding="utf-8") as source_file:
            source = source_file.read(MAX_SEARCH_FILE_BYTES + 1)
    except (OSError, UnicodeError):
        return None
    if len(source) > MAX_SEARCH_FILE_BYTES or "\x00" in source:
        return None
    try:
        tree = ast.parse(source, filename=display_path, type_comments=True)
    except (SyntaxError, ValueError):
        return None
    return _ParsedFile(path, display_path, source, tree)


def _source_lines(parsed: _ParsedFile, start: int, end: int) -> str:
    return "\n".join(parsed.lines[max(0, start - 1):end])


def _qualified_nodes(tree: ast.AST) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}

    def visit(node: ast.AST, prefix: tuple[str, ...]) -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name_parts = (*prefix, child.name)
                result[".".join(name_parts)] = child
                if isinstance(child, ast.ClassDef):
                    visit(child, name_parts)

    visit(tree, ())
    return result


def _signature(parsed: _ParsedFile, node: ast.AST) -> str:
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(item.lineno for item in decorators)])
    body = getattr(node, "body", [])
    end = max(start, body[0].lineno - 1) if body else node.end_lineno
    return _source_lines(parsed, start, end)


def _target_body(parsed: _ParsedFile, node: ast.AST) -> str:
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(item.lineno for item in decorators)])
    return _source_lines(parsed, start, node.end_lineno)


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    return names


def _caller_candidates(
    context: ToolUseContext,
    root: str,
    target_name: str,
    excluded_path: str,
) -> list[str]:
    results: list[str] = []
    scanned_bytes = 0
    for index, (path, display_path) in enumerate(_iter_project_files(context, root, root)):
        if index >= MAX_SEARCH_FILES or len(results) >= MAX_EDIT_CONTEXT_CALLERS:
            break
        if not display_path.endswith(".py") or os.path.realpath(display_path) == excluded_path:
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > MAX_SEARCH_FILE_BYTES or scanned_bytes + size > MAX_SEARCH_TOTAL_BYTES:
            continue
        scanned_bytes += size
        parsed = _read_python_file(path, display_path)
        if parsed is None:
            continue
        relative = _relative_display_path(root, display_path)
        for node in ast.walk(parsed.tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            call_name = (
                function.id if isinstance(function, ast.Name)
                else function.attr if isinstance(function, ast.Attribute)
                else None
            )
            if call_name != target_name:
                continue
            start = max(1, node.lineno - 3)
            end = min(len(parsed.lines), getattr(node, "end_lineno", node.lineno) + 3)
            results.append(f"{relative}:{node.lineno}\n{_source_lines(parsed, start, end)}")
            if len(results) >= MAX_EDIT_CONTEXT_CALLERS:
                break
    return results


def _jedi_references(
    parsed: _ParsedFile,
    target: ast.AST,
    context: ToolUseContext,
    root: str,
) -> list[str] | None:
    try:
        import jedi
    except ImportError:
        return None

    source_by_relative_path: dict[str, str] = {}
    scanned_bytes = 0
    for index, (path, display_path) in enumerate(_iter_project_files(context, root, root)):
        if index >= MAX_SEARCH_FILES:
            break
        if not display_path.endswith(".py") or _is_excluded_path(root, display_path):
            continue
        try:
            metadata = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_SEARCH_FILE_BYTES:
                continue
            if scanned_bytes + metadata.st_size > MAX_SEARCH_TOTAL_BYTES:
                break
            with open(path, "r", encoding="utf-8") as source_file:
                source = source_file.read(MAX_SEARCH_FILE_BYTES + 1)
        except (OSError, UnicodeError):
            continue
        if len(source) > MAX_SEARCH_FILE_BYTES or "\x00" in source:
            continue
        scanned_bytes += metadata.st_size
        relative = os.path.relpath(display_path, root)
        source_by_relative_path[relative] = source

    target_relative = os.path.relpath(parsed.display_path, root)
    source_by_relative_path[target_relative] = parsed.source
    references: list[str] = []
    with tempfile.TemporaryDirectory(prefix="agent-jedi-scope-") as snapshot_root:
        for relative, source in source_by_relative_path.items():
            destination = os.path.join(snapshot_root, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "w", encoding="utf-8") as output_file:
                output_file.write(source)

        project = jedi.Project(
            snapshot_root,
            added_sys_path=[snapshot_root],
            smart_sys_path=False,
        )
        script = jedi.Script(
            code=parsed.source,
            path=os.path.join(snapshot_root, target_relative),
            project=project,
        )
        keyword_length = 10 if isinstance(target, ast.AsyncFunctionDef) else 4
        try:
            names = script.get_references(
                line=target.lineno,
                column=target.col_offset + keyword_length,
                include_builtins=False,
            )
        except Exception:
            return None

        scope = getattr(context, "file_scope", None)
        allowed_paths = (
            {os.path.realpath(path) for path in scope.read_paths | scope.write_paths}
            if scope is not None else None
        )
        for name in names:
            module_path = getattr(name, "module_path", None)
            line_number = getattr(name, "line", None)
            column = getattr(name, "column", None)
            if module_path is None or line_number is None:
                continue
            try:
                relative = os.path.relpath(str(module_path), snapshot_root)
                if relative == os.pardir or relative.startswith(os.pardir + os.sep):
                    continue
                original_path = os.path.realpath(os.path.join(root, relative))
            except (OSError, ValueError):
                continue
            if _is_excluded_path(root, original_path):
                continue
            if allowed_paths is not None and original_path not in allowed_paths:
                continue
            source = source_by_relative_path.get(relative, "")
            lines = source.splitlines()
            if not 1 <= line_number <= len(lines):
                continue
            role = "definition" if name.is_definition() else "reference"
            relative_display = _relative_display_path(root, original_path)
            references.append(
                f"{relative_display}:{line_number}:{(column or 0) + 1} [{role}] "
                f"{lines[line_number - 1][:500]}"
            )
            if len(references) >= 100:
                break
    return references


def _fit_sections(sections: list[tuple[str, str]], token_budget: int) -> str:
    char_budget = token_budget * 4
    output: list[str] = []
    used = 0
    for title, body in sections:
        if not body:
            continue
        block = f"## {title}\n{body.strip()}\n"
        remaining = char_budget - used
        if remaining <= 0:
            output.append("[Context omitted to honor the token budget.]\n")
            break
        if len(block) > remaining:
            if remaining > 80:
                output.append(block[:remaining - 48] + "\n[Section truncated to honor budget.]\n")
            else:
                output.append("[Context omitted to honor the token budget.]\n")
            break
        output.append(block)
        used += len(block)
    return "\n".join(output)


def _file_skeleton(parsed: _ParsedFile, target: ast.AST) -> str:
    sections: list[str] = []
    for node in parsed.tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            sections.append(_source_lines(parsed, node.lineno, node.end_lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node is target:
                continue
            if isinstance(node, ast.ClassDef):
                class_lines = [_signature(parsed, node)]
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if member is target:
                            continue
                        class_lines.append(_signature(parsed, member))
                    elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                        class_lines.append(_source_lines(parsed, member.lineno, member.end_lineno))
                sections.append("\n".join(class_lines))
            else:
                sections.append(_signature(parsed, node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            sections.append(_source_lines(parsed, node.lineno, node.end_lineno))
    return "\n".join(sections)


def _build_edit_context(
    input: EditContextInput,
    context: ToolUseContext,
) -> str:
    root, requested_path = _resolve_search_path(context, input.file_path)
    if _is_excluded_path(root, requested_path):
        raise ValueError("GetEditContext cannot include generated or credential-like paths")
    if not requested_path.endswith(".py"):
        return "GetEditContext currently supports Python source files only."
    scope = getattr(context, "file_scope", None)
    path_error = scope.check_access("Read", requested_path) if scope is not None else None
    if path_error:
        raise ValueError(path_error)
    read_path = (
        scope.execution_path("Read", requested_path)
        if scope is not None and scope.staging_root else requested_path
    )
    parsed = _read_python_file(read_path, requested_path)
    if parsed is None:
        return (
            "Python parsing failed; no AST context package was produced. "
            "Fall back to Read for the complete file or a carefully chosen range."
        )

    nodes = _qualified_nodes(parsed.tree)
    matches = [
        (qualified_name, node) for qualified_name, node in nodes.items()
        if qualified_name == input.symbol or qualified_name.rsplit(".", 1)[-1] == input.symbol
    ]
    if len(matches) != 1:
        names = ", ".join(name for name, _node in matches[:20])
        if not matches:
            return f"No Python function or class named {input.symbol!r} was found."
        return f"Symbol {input.symbol!r} is ambiguous; use a qualified name: {names}"
    qualified_name, target = matches[0]
    if isinstance(target, ast.ClassDef):
        return "Select a method within the class; class-body context packages are not supported yet."

    target_body = _target_body(parsed, target)
    if len(target_body) > input.token_budget * 4 - 400:
        return (
            "The target function body exceeds the requested context budget. "
            "Increase token_budget or use Read with a suitable line range."
        )
    helper_names = sorted(_call_names(target) - {target.name})
    local_nodes = _qualified_nodes(parsed.tree)
    helpers: list[str] = []
    for helper_name in helper_names:
        candidates = [
            (name, node) for name, node in local_nodes.items()
            if name.rsplit(".", 1)[-1] == helper_name
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(candidates) != 1:
            continue
        name, node = candidates[0]
        if node.end_lineno - node.lineno + 1 <= MAX_EDIT_CONTEXT_HELPER_LINES:
            helpers.append(f"### {name}\n{_target_body(parsed, node)}")

    callers: list[str] = []
    if input.include_callers:
        callers = _caller_candidates(
            context, root, target.name, os.path.realpath(requested_path),
        )
    references = _jedi_references(parsed, target, context, root)

    package = _fit_sections([
        (f"Target {qualified_name} ({_relative_display_path(root, requested_path)})", target_body),
        ("File skeleton: imports, constants, and other signatures", _file_skeleton(parsed, target)),
        ("Short same-file helpers called by the target", "\n\n".join(helpers)),
        (
            "Jedi Python references (searched snapshot is limited to this context's file scope)",
            "\n".join(references or []),
        ),
        (
            "Additional possible callers (AST name matches only; dynamic dispatch is not covered)",
            "\n\n".join(callers),
        ),
    ], input.token_budget)
    approximate_tokens = max(1, (len(package) + 3) // 4)
    return (
        f"Python AST edit context for {qualified_name}. Approximate size: "
        f"{approximate_tokens} tokens (character estimate).\n"
        "Jedi references are Python semantic analysis, not compiler/type-checker guarantees. "
        "AST caller candidates are incomplete.\n\n"
        f"{package}"
    )


def _build_references(input: FindReferencesInput, context: ToolUseContext) -> str:
    root, requested_path = _resolve_search_path(context, input.file_path)
    if _is_excluded_path(root, requested_path):
        raise ValueError("FindReferences cannot inspect generated or credential-like paths")
    if not requested_path.endswith(".py"):
        return "FindReferences currently supports Python source files only."
    scope = getattr(context, "file_scope", None)
    path_error = scope.check_access("Read", requested_path) if scope is not None else None
    if path_error:
        raise ValueError(path_error)
    read_path = (
        scope.execution_path("Read", requested_path)
        if scope is not None and scope.staging_root else requested_path
    )
    parsed = _read_python_file(read_path, requested_path)
    if parsed is None:
        return "Could not parse a readable Python file."
    nodes = _qualified_nodes(parsed.tree)
    matches = [
        (name, node) for name, node in nodes.items()
        if name == input.symbol or name.rsplit(".", 1)[-1] == input.symbol
    ]
    if len(matches) != 1:
        if not matches:
            return f"No Python function or class named {input.symbol!r} was found."
        return "Symbol is ambiguous; use a qualified name: " + ", ".join(
            name for name, _node in matches[:20]
        )
    qualified_name, target = matches[0]
    if isinstance(target, ast.ClassDef):
        return "Select a method within the class to find references."
    references = _jedi_references(parsed, target, context, root)
    if references is None:
        return "Jedi is unavailable; use Grep for textual candidates instead."
    if not references:
        return f"Jedi found no references for {qualified_name} in the visible project scope."
    return (
        f"Jedi references for {qualified_name} (Python semantic analysis; not a type-checker guarantee):\n"
        + "\n".join(references)
    )


def _pyright_position(parsed: _ParsedFile, target: ast.AST) -> tuple[int, int]:
    line = parsed.lines[target.lineno - 1]
    prefix_size = target.col_offset + (
        10 if isinstance(target, ast.AsyncFunctionDef) else 4
    )
    prefix = line.encode("utf-8")[:prefix_size].decode("utf-8", errors="ignore")
    character = len(prefix.encode("utf-16-le")) // 2
    return target.lineno - 1, character


def _find_target_node(parsed: _ParsedFile, symbol: str) -> tuple[str, ast.AST] | None:
    matches = [
        (name, node) for name, node in _qualified_nodes(parsed.tree).items()
        if name == symbol or name.rsplit(".", 1)[-1] == symbol
    ]
    return matches[0] if len(matches) == 1 else None


async def _pyright_navigation(
    input: PyrightNavigationInput,
    context: Any,
    operation: str,
) -> str:
    root, requested_path = _resolve_search_path(context, input.file_path)
    if _is_excluded_path(root, requested_path):
        raise ValueError("Pyright cannot inspect generated or credential-like paths")
    if not requested_path.endswith(".py"):
        return "Pyright navigation currently supports Python source files only."

    scope = getattr(context, "file_scope", None)
    if scope is not None:
        path_error = scope.check_access("Read", requested_path)
        if path_error:
            raise ValueError(path_error)
    source_path = (
        scope.execution_path("Read", requested_path)
        if scope is not None and scope.staging_root else requested_path
    )
    parsed = _read_python_file(source_path, requested_path)
    if parsed is None:
        return "Could not parse a readable Python file."
    target_match = _find_target_node(parsed, input.symbol)
    if target_match is None:
        return f"No unique Python function named {input.symbol!r} was found."
    qualified_name, target = target_match
    if isinstance(target, ast.ClassDef):
        return "Select a method within the class for Pyright navigation."

    command = os.environ.get("AGENT_PYRIGHT_LANGSERVER") or shutil.which(
        "pyright-langserver",
    )
    if not command:
        return (
            "Pyright language server is unavailable. Install "
            "schema manger/requirements-code-intel.txt and Node.js."
        )

    temporary_workspace = tempfile.TemporaryDirectory(prefix="agent-pyright-scope-")
    workspace_root = temporary_workspace.name
    lsp_file_path = os.path.join(workspace_root, os.path.relpath(requested_path, root))

    try:
        copied_source = False
        visible_documents: list[tuple[str, str]] = []
        scanned_bytes = 0
        for path, display_path in _iter_project_files(context, root, root):
            if not display_path.endswith(".py"):
                continue
            try:
                size = os.path.getsize(path)
                if size > MAX_SEARCH_FILE_BYTES or scanned_bytes + size > MAX_SEARCH_TOTAL_BYTES:
                    continue
                with open(path, "rb") as source_file:
                    content = source_file.read(MAX_SEARCH_FILE_BYTES + 1)
            except OSError:
                continue
            if len(content) > MAX_SEARCH_FILE_BYTES or b"\x00" in content:
                continue
            scanned_bytes += len(content)
            relative = os.path.relpath(display_path, root)
            destination = os.path.join(workspace_root, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as target_file:
                target_file.write(content)
            text = content.decode("utf-8", errors="replace")
            visible_documents.append((destination, text))
            if os.path.realpath(display_path) == os.path.realpath(requested_path):
                copied_source = True
        if not copied_source:
            return "The target file is not present in the visible Pyright snapshot."

        line, character = _pyright_position(parsed, target)
        client = await JsonRpcStdioClient.start(workspace_root, command=command)
        try:
            for document_path, document_text in visible_documents:
                await client.open_document(document_path, document_text)
            if operation == "references":
                locations = await client.references(lsp_file_path, line, character)
            else:
                result = await client.definition(lsp_file_path, line, character)
                locations = result if isinstance(result, list) else ([result] if result else [])
        finally:
            await client.close()

        allowed_paths = (
            {os.path.realpath(path) for path in scope.read_paths | scope.write_paths}
            if scope is not None else None
        )
        rows = []
        for location in locations[:100]:
            uri = location.get("uri") or location.get("targetUri")
            range_value = location.get("range") or location.get("targetSelectionRange")
            if not uri or not range_value:
                continue
            parsed_uri = urlparse(uri)
            if parsed_uri.scheme != "file":
                continue
            lsp_path = os.path.realpath(unquote(parsed_uri.path))
            try:
                relative = os.path.relpath(lsp_path, workspace_root)
                if relative == os.pardir or relative.startswith(os.pardir + os.sep):
                    continue
            except ValueError:
                continue
            original_path = os.path.realpath(os.path.join(root, relative))
            if allowed_paths is not None and original_path not in allowed_paths:
                continue
            start = range_value.get("start", {})
            rows.append(
                f"{_relative_display_path(root, original_path)}:"
                f"{int(start.get('line', 0)) + 1}:{int(start.get('character', 0)) + 1}"
            )
        if not rows:
            return (
                f"Pyright found no {operation} results for {qualified_name} in the "
                "active workspace. This is not proof that the symbol is unused."
            )
        scope_label = "declared Task snapshot" if scope is not None else "project workspace"
        label = "references" if operation == "references" else "definition"
        return f"Pyright {label} for {qualified_name} in {scope_label}:\n" + "\n".join(rows)
    finally:
        if temporary_workspace is not None:
            temporary_workspace.cleanup()


async def _pyright_references(
    input: PyrightNavigationInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    try:
        return ToolResult(data=await _pyright_navigation(input, context, "references"))
    except Exception as error:
        return ToolResult(data=(
            f"Pyright reference request failed: {type(error).__name__}. "
            "Use Jedi or Grep as a separately labeled fallback."
        ))


async def _pyright_definition(
    input: PyrightNavigationInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    try:
        return ToolResult(data=await _pyright_navigation(input, context, "definition"))
    except Exception as error:
        return ToolResult(data=(
            f"Pyright definition request failed: {type(error).__name__}. "
            "Use Jedi or Grep as a separately labeled fallback."
        ))


async def _ast_grep(
    input: AstGrepInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    root, base_path = _resolve_search_path(context, input.path)
    binary = os.environ.get("AGENT_AST_GREP_BINARY") or os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "node_modules", ".bin", "ast-grep",
    )
    if not os.path.isfile(binary):
        binary = shutil.which("ast-grep") or ""
    if not binary:
        return ToolResult(data=(
            "ast-grep is unavailable. Run `npm install --prefix 'schema manger'` "
            "or set AGENT_AST_GREP_BINARY."
        ))

    files: list[tuple[str, str]] = []
    truncated = False
    for search_path, display_path in _iter_project_files(context, root, base_path):
        suffix_language = {
            ".py": "python", ".pyi": "python", ".ts": "typescript",
            ".tsx": "tsx", ".js": "javascript", ".jsx": "javascript",
            ".mjs": "javascript", ".go": "go", ".rs": "rust",
        }.get(os.path.splitext(display_path)[1].lower())
        if suffix_language != input.language:
            continue
        files.append((search_path, display_path))
        if len(files) > MAX_AST_GREP_FILES:
            truncated = True
            files.pop()
            break
    if not files:
        return ToolResult(data=f"No {input.language} files in the selected visible scope.")

    command = [
        binary, "run", "--pattern", input.pattern,
        "--lang", AST_GREP_LANGUAGES[input.language],
        "--json=stream", "--color", "never", "--threads", "1",
        *(path for path, _display in files),
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=MAX_AST_GREP_OUTPUT_BYTES,
        )
    except OSError as error:
        return ToolResult(data=f"Could not start ast-grep: {type(error).__name__}.")

    file_display = {
        os.path.realpath(search_path): display_path
        for search_path, display_path in files
    }
    matches: list[str] = []
    output_bytes = 0
    timed_out = False
    output_limited = False
    stderr_task = asyncio.create_task(process.stderr.read(8_192)) if process.stderr else None
    try:
        deadline = asyncio.get_running_loop().time() + AST_GREP_TIMEOUT_SECONDS
        while process.stdout is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                timed_out = True
                break
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                timed_out = True
                break
            if not line:
                break
            output_bytes += len(line)
            if output_bytes > MAX_AST_GREP_OUTPUT_BYTES:
                output_limited = True
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            path = str(record.get("file", ""))
            resolved_path = os.path.realpath(
                path if os.path.isabs(path) else os.path.join(root, path),
            )
            display_path = file_display.get(resolved_path)
            if display_path is None:
                continue
            location = record.get("range", {}).get("start", {})
            row = int(location.get("line", 0))
            column = int(location.get("column", 0)) + 1
            snippet = " ".join(str(record.get("text", "")).split())
            matches.append(
                f"{_relative_display_path(root, display_path)}:{row}:{column}: "
                f"{snippet[:500]}"
            )
            if len(matches) >= input.max_results:
                output_limited = True
                break
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
    stderr_text = ""
    if stderr_task is not None:
        try:
            stderr_text = (await stderr_task).decode("utf-8", errors="replace")[:1_000]
        except Exception:
            pass

    if process.returncode and not matches:
        detail = " ".join(stderr_text.split())
        suffix = f" {detail}" if detail else ""
        return ToolResult(data=f"ast-grep failed with exit code {process.returncode}.{suffix}")

    if not matches:
        output = f"No structural matches for {input.pattern!r}."
    else:
        output = "\n".join(matches)
    notices = []
    if truncated:
        notices.append(f"file scan capped at {MAX_AST_GREP_FILES} files")
    if output_limited:
        notices.append("match/output limit reached")
    if timed_out:
        notices.append(f"search stopped after {AST_GREP_TIMEOUT_SECONDS}s")
    if notices:
        output += "\n[Incomplete results: " + "; ".join(notices) + ".]"
    return ToolResult(data=output)


def _build_repo_map(context: ToolUseContext, path: str, token_budget: int) -> str:
    root, base_path = _resolve_search_path(context, path)
    parsed_files: list[_ParsedFile] = []
    scanned_bytes = 0
    truncated = False
    for index, (file_path, display_path) in enumerate(
        _iter_project_files(context, root, base_path),
    ):
        if index >= MAX_REPO_MAP_FILES or len(parsed_files) >= MAX_REPO_MAP_FILES:
            truncated = True
            break
        if not display_path.endswith(".py"):
            continue
        try:
            size = os.path.getsize(file_path)
        except OSError:
            continue
        if size > MAX_SEARCH_FILE_BYTES or scanned_bytes + size > MAX_SEARCH_TOTAL_BYTES:
            truncated = True
            break
        scanned_bytes += size
        parsed = _read_python_file(file_path, display_path)
        if parsed is not None:
            parsed_files.append(parsed)

    if not parsed_files:
        return "No readable Python modules were found in the selected scope."

    module_paths: dict[str, str] = {}
    path_modules: dict[str, str] = {}
    for parsed in parsed_files:
        relative = os.path.relpath(parsed.display_path, root).replace(os.sep, "/")
        parts = relative[:-3].split("/")
        if parts[-1] == "__init__":
            parts.pop()
        module_name = ".".join(part for part in parts if part and part != ".")
        module_paths[module_name] = parsed.display_path
        path_modules[parsed.display_path] = module_name

    edges: dict[str, set[str]] = {parsed.display_path: set() for parsed in parsed_files}
    for parsed in parsed_files:
        current_module = path_modules[parsed.display_path]
        for node in ast.walk(parsed.tree):
            if not isinstance(node, ast.ImportFrom):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name
                        if target in module_paths:
                            edges[parsed.display_path].add(module_paths[target])
                continue
            if node.level:
                current_parts = current_module.split(".")[:-1]
                keep = max(0, len(current_parts) - node.level + 1)
                module = ".".join(current_parts[:keep] + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            if module in module_paths:
                edges[parsed.display_path].add(module_paths[module])

    scores = {path: 1.0 / len(parsed_files) for path in edges}
    damping = 0.85
    for _iteration in range(12):
        updated = {path: (1.0 - damping) / len(parsed_files) for path in edges}
        for source_path, targets in edges.items():
            if not targets:
                share = damping * scores[source_path] / len(edges)
                for target_path in updated:
                    updated[target_path] += share
                continue
            share = damping * scores[source_path] / len(targets)
            for target_path in targets:
                updated[target_path] += share
        scores = updated

    entries: list[tuple[float, str]] = []
    for parsed in parsed_files:
        signatures: list[str] = []
        for node in parsed.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                signatures.append(_signature(parsed, node))
                if isinstance(node, ast.ClassDef):
                    signatures.extend(
                        _signature(parsed, child)
                        for child in node.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    )
        if signatures:
            relative = _relative_display_path(root, parsed.display_path)
            entries.append((scores.get(parsed.display_path, 0.0), f"# {relative}\n" + "\n".join(signatures)))
    entries.sort(key=lambda item: (-item[0], item[1]))

    char_budget = token_budget * 4
    output: list[str] = []
    used = 0
    for _score, entry in entries:
        block = entry + "\n"
        if used + len(block) > char_budget:
            truncated = True
            break
        output.append(block)
        used += len(block)
    if truncated:
        output.append("<repo map truncated to the requested approximate token budget>\n")
    if not output:
        return "No module signatures fit the requested token budget."
    return (
        "Python-only read-only repository map. Files are ranked by an approximate "
        "PageRank over local import edges; this is not a resolved call graph.\n\n"
        + "\n".join(output)
    )


async def _get_edit_context(
    input: EditContextInput,
    context: ToolUseContext,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    return ToolResult(data=_build_edit_context(input, context))


async def _repo_map(
    input: RepoMapInput,
    context: ToolUseContext,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    try:
        connection, summary = await asyncio.to_thread(
            build_code_index, context, input.path,
        )
    except Exception as error:
        return ToolResult(data=(
            f"Tree-sitter index unavailable ({type(error).__name__}); "
            + _build_repo_map(context, input.path, input.token_budget)
        ))
    try:
        ranked = ranked_symbols(connection, limit=1_000)
        if not ranked:
            return ToolResult(data=(
                "Tree-sitter found no symbols in supported languages. "
                f"Skipped {summary.skipped_files} files; parse warnings: "
                f"{summary.parse_warnings}.\n\n"
                + _build_repo_map(context, input.path, input.token_budget)
            ))
        char_budget = input.token_budget * 4
        lines = [
            "Tree-sitter symbol map ranked by local imports and static call-name candidates.",
            "Call edges are syntactic name matches, not resolved references.",
        ]
        used = sum(len(line) + 1 for line in lines)
        included_files: set[str] = set()
        for item in ranked:
            relative = _relative_display_path(_project_root(context), item["file_path"])
            entry = (
                f"{relative}:{item['start_line']} "
                f"[{item['kind']}] {item['signature']}"
            )
            if used + len(entry) + 1 > char_budget:
                lines.append("<repo map truncated to the requested approximate token budget>")
                break
            lines.append(entry)
            used += len(entry) + 1
            included_files.add(relative)
        lines.append(
            f"Indexed {summary.files_indexed} changed/new files; reused "
            f"{summary.files_reused}; parse warnings {summary.parse_warnings}; "
            f"skipped {summary.skipped_files}."
        )
        return ToolResult(data="\n".join(lines))
    finally:
        close_code_index(connection)


async def _index_code(
    input: IndexCodeInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    try:
        connection, summary = await asyncio.to_thread(
            build_code_index, context, input.path,
        )
    except Exception as error:
        return ToolResult(data=(
            f"Tree-sitter indexing failed: {type(error).__name__}. "
            "Install schema manger/requirements-code-intel.txt."
        ))
    try:
        scope_note = "Task-scoped in-memory index" if summary.scope_limited else "Private persistent index"
        warning = (
            " Some files had syntax errors; extracted symbols may be incomplete."
            if summary.parse_warnings else ""
        )
        return ToolResult(data=(
            f"{scope_note}: indexed {summary.files_indexed} changed/new files, "
            f"reused {summary.files_reused}, removed {summary.files_removed}; "
            f"{summary.symbols} symbols; {summary.import_edges} import edges; "
            f"{summary.call_candidates} static call "
            f"candidates; {summary.parse_warnings} parse warnings; "
            f"{summary.skipped_files} skipped files.{warning}"
        ))
    finally:
        close_code_index(connection)


async def _find_callers(
    input: SymbolQueryInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    try:
        connection, summary = await asyncio.to_thread(
            build_code_index, context, input.path,
        )
    except Exception as error:
        return ToolResult(data=(
            f"Tree-sitter caller index failed: {type(error).__name__}. "
            "Install schema manger/requirements-code-intel.txt."
        ))
    try:
        matches = find_callers(connection, input.symbol)
        if not matches:
            incomplete = (
                f" Index incomplete: {summary.parse_warnings} parse warnings, "
                f"{summary.skipped_files} skipped files."
                if summary.parse_warnings or summary.skipped_files else ""
            )
            return ToolResult(data=(
                f"No static call-name candidates for {input.symbol!r} in the "
                "indexed scope. This is not proof that the symbol is unused; "
                "dynamic dispatch and unsupported syntax may be missed."
                + incomplete
            ))
        lines = [
            f"Static call-name candidates for {input.symbol!r}; not a resolved call graph."
        ]
        lines.extend(
            f"{_relative_display_path(_project_root(context), item['file_path'])}:"
            f"{item['line']} {item['caller_qualified_name']} calls {item['target_name']}"
            for item in matches
        )
        if summary.parse_warnings or summary.skipped_files:
            lines.append(
                f"Index is incomplete: {summary.parse_warnings} parse warnings, "
                f"{summary.skipped_files} skipped files."
            )
        return ToolResult(data="\n".join(lines))
    finally:
        close_code_index(connection)


async def _find_callees(
    input: SymbolQueryInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    try:
        connection, summary = await asyncio.to_thread(
            build_code_index, context, input.path,
        )
    except Exception as error:
        return ToolResult(data=(
            f"Tree-sitter callee index failed: {type(error).__name__}. "
            "Install schema manger/requirements-code-intel.txt."
        ))
    try:
        matches = find_callees(connection, input.symbol)
        if not matches:
            incomplete = (
                f" Index incomplete: {summary.parse_warnings} parse warnings, "
                f"{summary.skipped_files} skipped files."
                if summary.parse_warnings or summary.skipped_files else ""
            )
            return ToolResult(data=f"No indexed call expressions for {input.symbol!r}.{incomplete}")
        lines = [
            f"Static call expressions in {input.symbol!r}; targets are name-match candidates."
        ]
        for item in matches:
            target = item["candidate_symbol"] or "unresolved"
            target_path = (
                _relative_display_path(_project_root(context), item["candidate_file"])
                if item["candidate_file"] else ""
            )
            lines.append(
                f"{_relative_display_path(_project_root(context), item['file_path'])}:"
                f"{item['line']} {item['target_name']} -> {target} {target_path}"
            )
        if summary.parse_warnings or summary.skipped_files:
            lines.append(
                f"Index is incomplete: {summary.parse_warnings} parse warnings, "
                f"{summary.skipped_files} skipped files."
            )
        return ToolResult(data="\n".join(lines))
    finally:
        close_code_index(connection)


async def _find_imports(
    input: FileQueryInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    root, file_path = _resolve_search_path(context, input.file_path)
    if _is_excluded_path(root, file_path):
        raise ValueError("FindImports cannot inspect generated or credential-like paths")
    scope = getattr(context, "file_scope", None)
    if scope is not None:
        path_error = scope.check_access("Read", file_path)
        if path_error:
            raise ValueError(path_error)
    try:
        connection, summary = await asyncio.to_thread(
            build_code_index, context, input.file_path,
        )
    except Exception as error:
        return ToolResult(data=f"Tree-Sitter import index failed: {type(error).__name__}.")
    try:
        imports = find_imports(connection, file_path)
        if not imports:
            message = f"No import statements found for {_relative_display_path(root, file_path)}."
            if summary.parse_warnings or summary.skipped_files:
                message += (
                    f" Index incomplete: {summary.parse_warnings} parse warnings, "
                    f"{summary.skipped_files} skipped files."
                )
            return ToolResult(data=message)
        lines = [
            f"Syntax-level imports in {_relative_display_path(root, file_path)}; module resolution is not guaranteed."
        ]
        lines.extend(
            f"{item['line']}: {item['imported_module']}" for item in imports
        )
        if summary.parse_warnings or summary.skipped_files:
            lines.append(
                f"Index is incomplete: {summary.parse_warnings} parse warnings, "
                f"{summary.skipped_files} skipped files."
            )
        return ToolResult(data="\n".join(lines))
    finally:
        close_code_index(connection)


async def _find_references(
    input: FindReferencesInput,
    context: ToolUseContext,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    return ToolResult(data=_build_references(input, context))


async def _allow_read(input: BaseModel, _context: ToolUseContext):
    return PermissionAllowDecision(updated_input=input.model_dump())


def _map_result(data: str, tool_use_id: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": data}


GetEditContextTool = build_tool(
    name="GetEditContext",
    description=(
        "Build a budgeted Python AST context package for a function: full target body, "
        "file skeleton, short same-file helpers, and possible static callers."
    ),
    prompt=(
        "Use before editing a Python function when a full file read is unnecessary. "
        "Caller matches are syntactic candidates, not complete or compiler-resolved references."
    ),
    input_schema=EditContextInput,
    call=_get_edit_context,
    max_result_size_chars=MAX_EDIT_CONTEXT_TOKENS * 4 + 1_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: (
        f"GetEditContext({input.get('symbol', '')})"
    ),
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "file_path", "."),
)

RepoMapTool = build_tool(
    name="RepoMap",
    description=(
        "Build a token-bounded symbol map for Tree-sitter-supported languages, "
        "ranked by static call-name candidates."
    ),
    prompt=(
        "Use to orient in a project without reading every file. Tree-sitter maps "
        "syntax, not types; call edges are candidate name matches. The map is "
        "scope-limited and may be truncated to the budget."
    ),
    input_schema=RepoMapInput,
    call=_repo_map,
    max_result_size_chars=17_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"RepoMap({input.get('path', '.')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": True,
    },
    get_path=lambda input: getattr(input, "path", "."),
)

FindReferencesTool = build_tool(
    name="FindReferences",
    description=(
        "Find Python symbol definitions and references with Jedi, limited to a "
        "snapshot of the current project or declared Task file scope."
    ),
    prompt=(
        "Use before changing a Python function signature. Results are scoped and "
        "Jedi-based, but are not compiler or dynamic-dispatch guarantees."
    ),
    input_schema=FindReferencesInput,
    call=_find_references,
    max_result_size_chars=20_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: (
        f"FindReferences({input.get('symbol', '')})"
    ),
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "file_path", "."),
)

IndexCodeTool = build_tool(
    name="IndexCode",
    description=(
        "Incrementally index supported project source files with Tree-sitter into "
        "a private SQLite symbol/call-candidate database."
    ),
    prompt=(
        "Run when a project needs repeated code navigation. Task-scoped indexing "
        "stays in memory; parent project indexes are private to the local user. "
        "Syntax errors and skipped files are reported."
    ),
    input_schema=IndexCodeInput,
    call=_index_code,
    max_result_size_chars=4_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"IndexCode({input.get('path', '.')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: False,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "path", "."),
)

FindCallersTool = build_tool(
    name="FindCallers",
    description=(
        "Find Tree-sitter static call-name candidates for a symbol. Results do "
        "not prove completeness or safe deletion."
    ),
    prompt=(
        "Use before changing a signature as an additional candidate search. "
        "These edges are syntactic name matches, not resolved references."
    ),
    input_schema=SymbolQueryInput,
    call=_find_callers,
    max_result_size_chars=20_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"FindCallers({input.get('symbol', '')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "path", "."),
)

FindCalleesTool = build_tool(
    name="FindCallees",
    description="List Tree-sitter call expressions and possible target symbols for a function.",
    prompt="Call targets are syntactic name matches and may be ambiguous or unresolved.",
    input_schema=SymbolQueryInput,
    call=_find_callees,
    max_result_size_chars=20_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"FindCallees({input.get('symbol', '')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "path", "."),
)

FindImportsTool = build_tool(
    name="FindImports",
    description="List syntax-level imports from a file supported by the Tree-Sitter index.",
    prompt="Use to inspect a file's imports. Returned module names are syntax-level, not resolved dependencies.",
    input_schema=FileQueryInput,
    call=_find_imports,
    max_result_size_chars=8_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"FindImports({input.get('file_path', '')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "file_path", "."),
)

AstGrepTool = build_tool(
    name="AstGrep",
    description=(
        "Search visible project files for syntax-aware code patterns using the "
        "locally installed ast-grep CLI. This tool is read-only."
    ),
    prompt=(
        "Use structural patterns such as `def $NAME($$$)` or `function $F($$$) { $$$ }`. "
        "Choose the source language. Searches are bounded and Task workers are limited "
        "to their declared files. This tool does not rewrite files."
    ),
    input_schema=AstGrepInput,
    call=_ast_grep,
    max_result_size_chars=16_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"AstGrep({input.get('pattern', '')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "path", "."),
)

PyrightReferencesTool = build_tool(
    name="PyrightReferences",
    description=(
        "Find Python symbol references using the Pyright Language Server over stdio. "
        "Task workers are served from a declared-file snapshot."
    ),
    prompt=(
        "Use PyrightReferences before Python signature changes. Results are type-aware "
        "language-server results, but still depend on project configuration and visible files."
    ),
    input_schema=PyrightNavigationInput,
    call=_pyright_references,
    max_result_size_chars=20_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: (
        f"PyrightReferences({input.get('symbol', '')})"
    ),
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: False,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "file_path", "."),
)

PyrightDefinitionTool = build_tool(
    name="PyrightDefinition",
    description="Resolve a Python symbol definition with the Pyright Language Server.",
    prompt="Use for type-aware Python go-to-definition in the active project or declared Task snapshot.",
    input_schema=PyrightNavigationInput,
    call=_pyright_definition,
    max_result_size_chars=8_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: (
        f"PyrightDefinition({input.get('symbol', '')})"
    ),
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: False,
    check_permissions=_allow_read,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "file_path", "."),
)


__all__ = [
    "EditContextInput", "RepoMapInput", "FindReferencesInput",
    "IndexCodeInput", "SymbolQueryInput", "GetEditContextTool", "RepoMapTool",
    "FindReferencesTool", "IndexCodeTool", "FindCallersTool", "FindCalleesTool",
    "PyrightReferencesTool", "PyrightDefinitionTool", "AstGrepInput", "AstGrepTool",
    "FileQueryInput", "FindImportsInput", "FindImportsTool",
]
