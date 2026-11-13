"""Offline tests for project search tools and Task scope enforcement."""
import asyncio
import os
import tempfile

from execute import execute_tool
from registry import clear, register
from state import AppState, FileStateCache, ToolPermissionContext
from task_scheduler import FileLeaseRegistry, TaskFileScope
from tool import ToolUseContext
from tools.search import GrepInput, GrepTool, GlobInput, GlobTool


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
    if scope is not None:
        context.allowed_tool_names = frozenset({"Grep", "Glob"})
    return context


async def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        source_dir = os.path.join(root, "src")
        nested_dir = os.path.join(source_dir, "nested")
        docs_dir = os.path.join(root, "docs")
        os.makedirs(nested_dir)
        os.makedirs(docs_dir)
        source_path = os.path.join(source_dir, "alpha.py")
        nested_path = os.path.join(nested_dir, "beta.py")
        docs_path = os.path.join(docs_dir, "readme.md")
        secret_path = os.path.join(root, ".env")
        with open(source_path, "w", encoding="utf-8") as target:
            target.write("def alpha():\n    return 'Needle'\n")
        with open(nested_path, "w", encoding="utf-8") as target:
            target.write("# another needle\n")
        with open(docs_path, "w", encoding="utf-8") as target:
            target.write("needle in docs\n")
        with open(secret_path, "w", encoding="utf-8") as target:
            target.write("needle=do-not-return\n")

        context = make_context(root)
        grep_result = await GrepTool.call(
            GrepInput(pattern="needle", path="src"), context,
        )
        assert "src/alpha.py:2" in grep_result.data
        assert "src/nested/beta.py:1" in grep_result.data
        assert "docs/readme.md" not in grep_result.data
        print("✅ Literal Grep and path narrowing OK")

        glob_result = await GlobTool.call(
            GlobInput(pattern="**/*.py", path="."), context,
        )
        assert "src/alpha.py" in glob_result.data
        assert "src/nested/beta.py" in glob_result.data
        assert ".env" not in glob_result.data
        print("✅ Glob and sensitive-path filtering OK")

        leases = FileLeaseRegistry(root)
        scoped_path = os.path.realpath(source_path)
        scope = TaskFileScope(
            task_id="reader",
            project_root=os.path.realpath(root),
            read_paths=frozenset({scoped_path}),
            write_paths=frozenset(),
            leases=leases,
        )
        worker_context = make_context(root, scope)
        scoped_result = await GrepTool.call(
            GrepInput(pattern="needle", path="."), worker_context,
        )
        assert "src/alpha.py:2" in scoped_result.data
        assert "src/nested/beta.py" not in scoped_result.data
        assert "docs/readme.md" not in scoped_result.data
        print("✅ Worker search restricted to declared files")

        clear()
        register(GrepTool)
        denied = await execute_tool({
            "id": "scoped-search-denied",
            "name": "Grep",
            "input": {"pattern": "needle", "path": "docs"},
        }, worker_context)
        assert denied.is_error is True
        assert "task scope" in denied.result_message.content.lower()

        allowed = await execute_tool({
            "id": "scoped-search-allowed",
            "name": "Grep",
            "input": {"pattern": "needle", "path": "."},
        }, worker_context)
        assert allowed.is_error is False
        assert "src/alpha.py:2" in allowed.result_message.content
        assert "docs/readme.md" not in allowed.result_message.content
        print("✅ Executor enforces worker search scope")

        try:
            await GlobTool.call(GlobInput(pattern="*", path=".."), context)
            raise AssertionError("search path outside project should fail")
        except ValueError as error:
            assert "outside the project" in str(error)
        print("✅ Search cannot escape project root")

    clear()
    print("\n🎉 All search tool tests passed")


asyncio.run(main())