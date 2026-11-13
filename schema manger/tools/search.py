"""Project-local content and filename search tools."""
from __future__ import annotations

import fnmatch
import os
import stat
from typing import Any, Iterator

from pydantic import BaseModel, Field, field_validator

from permission_types import PermissionAllowDecision
from tool import ToolResult, ToolUseContext, build_tool


MAX_SEARCH_FILES = 10_000
MAX_SEARCH_FILE_BYTES = 1_048_576
MAX_SEARCH_TOTAL_BYTES = 32 * 1024 * 1024
MAX_SEARCH_RESULT_CHARS = 20_000
EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build",
})
SENSITIVE_PATH_PARTS = frozenset({".aws", ".gnupg", ".ssh", "credentials"})


class GrepInput(BaseModel):
    pattern: str = Field(min_length=1, max_length=512)
    path: str = Field(default=".", max_length=4096)
    max_results: int = Field(default=50, ge=1, le=100)
    case_sensitive: bool = False

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Search pattern must not be empty")
        if "\n" in value or "\r" in value:
            raise ValueError("Search pattern must fit on one line")
        return value


class GlobInput(BaseModel):
    pattern: str = Field(min_length=1, max_length=512)
    path: str = Field(default=".", max_length=4096)
    max_results: int = Field(default=200, ge=1, le=500)

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if os.path.isabs(value) or any(part == ".." for part in normalized.split("/")):
            raise ValueError("Glob pattern must stay within the selected project path")
        return value


def _project_root(context: ToolUseContext) -> str:
    return os.path.realpath(os.path.abspath(
        getattr(context, "project_root", None) or os.getcwd(),
    ))


def _resolve_search_path(context: ToolUseContext, requested_path: str) -> tuple[str, str]:
    root = _project_root(context)
    expanded = os.path.expanduser(requested_path or ".")
    candidate = expanded if os.path.isabs(expanded) else os.path.join(root, expanded)
    candidate = os.path.realpath(os.path.abspath(candidate))
    try:
        if os.path.commonpath((root, candidate)) != root:
            raise ValueError("Search path is outside the project")
    except ValueError as error:
        if str(error) == "Search path is outside the project":
            raise
        raise ValueError("Search path is outside the project") from error
    return root, candidate


def _is_excluded_path(root: str, display_path: str) -> bool:
    relative = os.path.relpath(display_path, root)
    parts = relative.replace("\\", "/").split("/")
    lowered = [part.lower() for part in parts]
    basename = lowered[-1]
    if any(part in EXCLUDED_DIRECTORIES for part in lowered):
        return True
    if any(part in SENSITIVE_PATH_PARTS for part in lowered):
        return True
    if basename.startswith(".env") or "secret" in basename or "credential" in basename:
        return True
    return basename.endswith((".pem", ".key", ".p12", ".pfx"))


def _iter_project_files(
    context: ToolUseContext,
    root: str,
    base_path: str,
) -> Iterator[tuple[str, str]]:
    scope = getattr(context, "file_scope", None)
    if scope is not None:
        declared_paths = scope.search_files(base_path)
        for display_path in declared_paths:
            if _is_excluded_path(root, display_path):
                continue
            search_path = (
                scope.execution_path("Read", display_path)
                if scope.staging_root else display_path
            )
            if os.path.isfile(search_path) and not os.path.islink(search_path):
                yield search_path, display_path
        return

    if os.path.isfile(base_path):
        if not os.path.islink(base_path) and not _is_excluded_path(root, base_path):
            yield base_path, base_path
        return
    if not os.path.isdir(base_path):
        return

    for current, directories, filenames in os.walk(base_path, followlinks=False):
        directories[:] = sorted(
            directory for directory in directories
            if directory not in EXCLUDED_DIRECTORIES
            and not os.path.islink(os.path.join(current, directory))
        )
        for filename in sorted(filenames):
            display_path = os.path.join(current, filename)
            if os.path.islink(display_path) or _is_excluded_path(root, display_path):
                continue
            yield display_path, display_path


def _relative_display_path(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


async def _grep(
    input: GrepInput,
    context: ToolUseContext,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    root, base_path = _resolve_search_path(context, input.path)
    needle = input.pattern if input.case_sensitive else input.pattern.casefold()
    results: list[str] = []
    scanned_bytes = 0
    truncated = False

    for file_index, (search_path, display_path) in enumerate(
        _iter_project_files(context, root, base_path),
    ):
        if file_index >= MAX_SEARCH_FILES:
            truncated = True
            break
        try:
            file_stat = os.stat(search_path, follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            if file_stat.st_size > MAX_SEARCH_FILE_BYTES:
                continue
            if scanned_bytes + file_stat.st_size > MAX_SEARCH_TOTAL_BYTES:
                truncated = True
                break
            with open(search_path, "rb") as source:
                content = source.read(MAX_SEARCH_FILE_BYTES + 1)
        except OSError:
            continue

        scanned_bytes += len(content)
        if len(content) > MAX_SEARCH_FILE_BYTES or b"\x00" in content:
            continue
        text = content.decode("utf-8", errors="replace")
        relative = _relative_display_path(root, display_path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            candidate = line if input.case_sensitive else line.casefold()
            if needle not in candidate:
                continue
            results.append(f"{relative}:{line_number}: {line[:1000]}")
            if len(results) >= input.max_results:
                truncated = True
                break
            if sum(len(item) + 1 for item in results) >= MAX_SEARCH_RESULT_CHARS:
                truncated = True
                break
        if truncated:
            break

    if not results:
        output = f"No matches for {input.pattern!r}."
    else:
        output = "\n".join(results)
    if truncated:
        output += "\n[Search output or scan limit reached.]"
    if len(output) > MAX_SEARCH_RESULT_CHARS:
        output = output[:MAX_SEARCH_RESULT_CHARS] + "\n[Output truncated.]"
    return ToolResult(data=output)


async def _glob(
    input: GlobInput,
    context: ToolUseContext,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    root, base_path = _resolve_search_path(context, input.path)
    pattern = input.pattern.replace("\\", "/")
    matches: list[str] = []
    truncated = False

    for file_index, (_search_path, display_path) in enumerate(
        _iter_project_files(context, root, base_path),
    ):
        if file_index >= MAX_SEARCH_FILES:
            truncated = True
            break
        relative_to_base = os.path.relpath(display_path, base_path).replace(os.sep, "/")
        matched = fnmatch.fnmatchcase(relative_to_base, pattern)
        if not matched and pattern.startswith("**/"):
            matched = fnmatch.fnmatchcase(relative_to_base, pattern[3:])
        if not matched:
            continue
        matches.append(_relative_display_path(root, display_path))
        if len(matches) >= input.max_results:
            truncated = True
            break

    output = "\n".join(matches) if matches else f"No files match {input.pattern!r}."
    if truncated:
        output += "\n[Search output or scan limit reached.]"
    return ToolResult(data=output)


async def _allow_search(input: BaseModel, _context: ToolUseContext):
    return PermissionAllowDecision(updated_input=input.model_dump())


def _map_result(data: str, tool_use_id: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": data}


GrepTool = build_tool(
    name="Grep",
    description=(
        "Search project files for literal text. Searches are bounded and skip "
        "generated directories and credential-like paths."
    ),
    prompt=(
        "Search project files for a literal text pattern. Matching is "
        "case-insensitive by default; set case_sensitive when needed. "
        "Search stays within the project and Task workers are limited to their "
        "declared file scopes."
    ),
    input_schema=GrepInput,
    call=_grep,
    max_result_size_chars=MAX_SEARCH_RESULT_CHARS,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"Grep({input.get('pattern', '')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_search,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": True, "isList": False,
    },
    get_path=lambda input: getattr(input, "path", "."),
)

GlobTool = build_tool(
    name="Glob",
    description=(
        "List project files matching a glob pattern. Results stay within the "
        "project and declared Task file scopes."
    ),
    prompt=(
        "Find project files by relative path pattern, such as **/*.py. "
        "Search skips generated directories and credential-like paths. "
        "Task workers can see only their declared file scopes."
    ),
    input_schema=GlobInput,
    call=_glob,
    max_result_size_chars=MAX_SEARCH_RESULT_CHARS,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: f"Glob({input.get('pattern', '')})",
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_allow_search,
    is_search_or_read_command=lambda _input: {
        "isSearch": True, "isRead": False, "isList": True,
    },
    get_path=lambda input: getattr(input, "path", "."),
)


__all__ = ["GrepInput", "GlobInput", "GrepTool", "GlobTool"]