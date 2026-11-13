"""Offline tests for the Python AST context package and repo map."""
import asyncio
import os
import tempfile

from execute import execute_tool
from registry import clear, register
from state import AppState, FileStateCache, ToolPermissionContext
from task_scheduler import FileLeaseRegistry, TaskFileScope
from tool import ToolUseContext
from tools.code_intelligence import (
    EditContextInput,
    AstGrepInput,
    AstGrepTool,
    FindReferencesInput,
    FindReferencesTool,
    FindImportsInput,
    FindImportsTool,
    GetEditContextTool,
    PyrightDefinitionTool,
    PyrightNavigationInput,
    PyrightReferencesTool,
    RepoMapInput,
    RepoMapTool,
)


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
        context.allowed_tool_names = frozenset({
            "GetEditContext", "RepoMap", "FindReferences", "FindCallers",
            "FindCallees", "PyrightReferences", "PyrightDefinition", "AstGrep",
        })
    return context


async def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        source_dir = os.path.join(root, "src")
        os.makedirs(source_dir)
        helper_path = os.path.join(source_dir, "helpers.py")
        app_path = os.path.join(source_dir, "app.py")
        secret_path = os.path.join(root, ".env")
        large_path = os.path.join(source_dir, "large.py")
        broken_path = os.path.join(source_dir, "broken.py")
        with open(helper_path, "w", encoding="utf-8") as target:
            target.write(
                "from typing import Iterable\n\n"
                "def normalize(items: Iterable[int]) -> list[int]:\n"
                "    return [int(item) for item in items]\n\n"
                "class Ledger:\n"
                "    label = 'ledger'\n\n"
                "    def total(self, items: Iterable[int]) -> int:\n"
                "        values = normalize(items)\n"
                "        return sum(values)\n"
            )
        with open(app_path, "w", encoding="utf-8") as target:
            target.write(
                "from src.helpers import Ledger\n\n"
                "def run(ledger: Ledger) -> int:\n"
                "    return ledger.total([1, 2, 3])\n"
            )
        with open(secret_path, "w", encoding="utf-8") as target:
            target.write("TOKEN=do-not-read\n")
        with open(large_path, "w", encoding="utf-8") as target:
            target.write("def huge():\n    return '" + ("x" * 2_000) + "'\n")
        with open(broken_path, "w", encoding="utf-8") as target:
            target.write("def broken(:\n    return 3\n")

        context = make_context(root)
        edit_result = await GetEditContextTool.call(
            EditContextInput(
                file_path=helper_path,
                symbol="Ledger.total",
                token_budget=600,
            ),
            context,
        )
        package = edit_result.data
        assert "def total(self, items: Iterable[int]) -> int:" in package
        assert "def normalize(items: Iterable[int]) -> list[int]:" in package
        assert "return [int(item) for item in items]" in package
        assert "src/app.py" in package
        assert "ledger.total([1, 2, 3])" in package
        assert "not compiler/type-checker guarantees" in package
        assert len(package) <= 600 * 4 + 500
        print("✅ AST edit context includes target, skeleton, helper, and candidates")

        references = await FindReferencesTool.call(
            FindReferencesInput(file_path=helper_path, symbol="total"),
            context,
        )
        assert "src/helpers.py:9" in references.data
        assert "[definition]" in references.data
        assert "src/app.py:4" in references.data
        assert "[reference]" in references.data
        print("✅ Jedi resolves Python references across project files")

        pyright_references = await PyrightReferencesTool.call(
            PyrightNavigationInput(file_path=helper_path, symbol="total"),
            context,
        )
        assert "src/helpers.py:9" in pyright_references.data
        assert "src/app.py:4" in pyright_references.data, pyright_references.data
        pyright_definition = await PyrightDefinitionTool.call(
            PyrightNavigationInput(file_path=app_path, symbol="run"),
            context,
        )
        assert "src/app.py:3" in pyright_definition.data
        print("✅ Pyright tools resolve references and definitions")

        structural = await AstGrepTool.call(
            AstGrepInput(
                pattern="def $NAME($$$)",
                language="python",
                path="src",
                max_results=10,
            ),
            context,
        )
        assert "src/helpers.py" in structural.data
        assert "normalize" in structural.data
        assert "run" in structural.data
        print("✅ ast-grep performs read-only structural search")

        map_result = await RepoMapTool.call(
            RepoMapInput(path=".", token_budget=512), context,
        )
        repo_map = map_result.data
        assert "src/helpers.py" in repo_map
        assert "class Ledger:" in repo_map
        assert "def run(ledger: Ledger) -> int:" in repo_map
        assert "Tree-sitter symbol map ranked by local imports and static call-name candidates" in repo_map
        assert len(repo_map) <= 512 * 4 + 500
        imports = await FindImportsTool.call(
            FindImportsInput(file_path=app_path), context,
        )
        assert "src.helpers" in imports.data
        print("✅ Import-ranked repo map and FindImports respect the budget/scope")

        try:
            await GetEditContextTool.call(
                EditContextInput(
                    file_path=secret_path,
                    symbol="missing",
                    token_budget=256,
                ),
                context,
            )
            raise AssertionError("credential-like context should be denied")
        except ValueError as error:
            assert "credential-like" in str(error)
        print("✅ AST context excludes credential-like files")

        scope = TaskFileScope(
            task_id="reader",
            project_root=os.path.realpath(root),
            read_paths=frozenset({os.path.realpath(helper_path)}),
            write_paths=frozenset(),
            leases=FileLeaseRegistry(root),
        )
        worker_context = make_context(root, scope)
        scoped_map = await RepoMapTool.call(
            RepoMapInput(path=".", token_budget=512), worker_context,
        )
        assert "src/helpers.py" in scoped_map.data
        assert "src/app.py" not in scoped_map.data
        scoped_references = await FindReferencesTool.call(
            FindReferencesInput(file_path=helper_path, symbol="total"),
            worker_context,
        )
        assert "[definition]" in scoped_references.data
        assert "src/app.py" not in scoped_references.data
        scoped_pyright = await PyrightReferencesTool.call(
            PyrightNavigationInput(file_path=helper_path, symbol="total"),
            worker_context,
        )
        assert "src/helpers.py:9" in scoped_pyright.data
        assert "src/app.py" not in scoped_pyright.data
        scoped_structural = await AstGrepTool.call(
            AstGrepInput(
                pattern="def $NAME($$$)",
                language="python",
                path=".",
                max_results=10,
            ),
            worker_context,
        )
        assert "src/helpers.py" in scoped_structural.data
        assert "src/app.py" not in scoped_structural.data
        print("✅ Repo map stays within the worker manifest")

        clear()
        register(GetEditContextTool)
        denied = await execute_tool({
            "id": "edit-context-outside",
            "name": "GetEditContext",
            "input": {
                "file_path": app_path,
                "symbol": "run",
                "token_budget": 512,
            },
        }, worker_context)
        assert denied.is_error
        assert "not permitted to read" in denied.result_message.content
        print("✅ Executor rejects out-of-manifest edit context")

        oversized = await GetEditContextTool.call(
            EditContextInput(
                file_path=large_path,
                symbol="huge",
                token_budget=256,
            ),
            context,
        )
        assert "exceeds the requested context budget" in oversized.data
        print("✅ Edit context handles target budget boundary")

        parse_failure = await GetEditContextTool.call(
            EditContextInput(
                file_path=broken_path,
                symbol="broken",
                token_budget=256,
            ),
            context,
        )
        assert "Python parsing failed" in parse_failure.data
        assert "Fall back to Read" in parse_failure.data
        imports_failure = await FindImportsTool.call(
            FindImportsInput(file_path=broken_path),
            context,
        )
        assert "Index incomplete" in imports_failure.data
        assert "parse warnings" in imports_failure.data
        print("✅ Parse failures are explicit and suggest Read fallback")

    clear()
    print("\n🎉 All code-intelligence tests passed")


asyncio.run(main())
