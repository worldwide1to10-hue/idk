"""Offline MCP lifecycle and permission tests using a fake stdio server."""
import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from types import SimpleNamespace

import mcp_runtime
from execute import execute_tool
from query import _tools_for_context
from registry import clear, find, register
from state import AppState, FileStateCache, ToolPermissionContext
from tool import ToolUseContext


class FakeSession:
    def __init__(self, _read_stream, _write_stream):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name="create_issue",
                description="Create an issue in the project tracker.",
                inputSchema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            ),
            SimpleNamespace(
                name="search_issues",
                description="Search issues in the project tracker.",
                inputSchema={"type": "object", "properties": {}},
            ),
            SimpleNamespace(
                name="list_projects",
                description="List available workspaces.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(text="created")],
            structuredContent={"ok": True},
            isError=False,
        )


def make_context(root: str) -> ToolUseContext:
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    return ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda _update: None,
        read_file_state=state.read_file_cache,
        project_root=root,
    )


async def main() -> None:
    try:
        mcp_runtime.MCPManager._validate_tool_schema({
            "type": "object",
            "properties": {"value": {"$ref": "https://example.invalid/schema"}},
        })
        raise AssertionError("remote schema references should be rejected")
    except ValueError as error:
        assert "Remote MCP schema references" in str(error)

    original_stdio_client = mcp_runtime.stdio_client
    original_client_session = mcp_runtime.ClientSession
    fake_sessions = []

    @asynccontextmanager
    async def fake_stdio_client(_params):
        yield object(), object()

    def fake_client_session(read_stream, write_stream):
        session = FakeSession(read_stream, write_stream)
        fake_sessions.append(session)
        return session

    mcp_runtime.stdio_client = fake_stdio_client
    mcp_runtime.ClientSession = fake_client_session
    clear()
    try:
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, "mcp.json")
            with open(config_path, "w", encoding="utf-8") as target:
                json.dump({
                    "servers": {
                        "tracker": {
                            "description": "Issue tracking and pull requests.",
                            "command": "trusted-server",
                            "args": ["--stdio"],
                        },
                        "calendar": {
                            "description": "Calendar and scheduling.",
                            "command": "unused-server",
                            "args": [],
                        },
                    },
                }, target)

            os.environ["MCP_TEST_TOKEN"] = "test-token"
            assert mcp_runtime.MCPManager._server_environment({
                "TRACKER_TOKEN": "${MCP_TEST_TOKEN}",
            })["TRACKER_TOKEN"] == "test-token"
            assert "MCP_TEST_TOKEN" not in mcp_runtime.MCPManager._server_environment({})
            del os.environ["MCP_TEST_TOKEN"]

            context = make_context(root)
            manager = mcp_runtime.MCPManager(config_path)
            context.mcp_manager = manager
            assert not manager.connected
            assert not manager._tools
            assert _tools_for_context(context) == []
            print("✅ MCP servers stay stopped until discovery")

            search_result = await manager.search_and_activate("create issue", context)
            assert manager.connected
            assert len(fake_sessions) == 1
            assert len(manager._tools) == 3
            assert "mcp__tracker__create_issue" in search_result
            assert context.active_mcp_tool_names == {
                "mcp__tracker__create_issue",
                "mcp__tracker__search_issues",
            }
            exposed = _tools_for_context(context)
            assert {tool.name for tool in exposed} == context.active_mcp_tool_names
            print("✅ ToolSearch activates only matching schema")

            denied = await execute_tool({
                "id": "mcp-denied",
                "name": "mcp__tracker__create_issue",
                "input": {"title": "Review request"},
            }, context, can_use_tool=lambda _decision: asyncio.sleep(0, result=False))
            assert denied.is_error
            assert not fake_sessions[0].calls
            print("✅ MCP call waits for explicit user approval")

            approved = await execute_tool({
                "id": "mcp-approved",
                "name": "mcp__tracker__create_issue",
                "input": {"title": "Review request"},
            }, context, can_use_tool=lambda _decision: asyncio.sleep(0, result=True))
            assert not approved.is_error
            assert "Untrusted data from MCP server" in approved.result_message.content
            assert fake_sessions[0].calls == [
                ("create_issue", {"title": "Review request"}),
            ]
            print("✅ MCP results are bounded and marked untrusted")

            inactive = await execute_tool({
                "id": "mcp-inactive",
                "name": "mcp__tracker__list_projects",
                "input": {},
            }, context)
            assert inactive.is_error
            assert "not been activated" in inactive.result_message.content
            print("✅ Unactivated MCP tools cannot be invoked")

            tool = find("mcp__tracker__create_issue")
            await manager.close()
            assert not manager.connected
            assert find("mcp__tracker__create_issue") is None
            assert tool is not None
            print("✅ MCP sessions close and unregister tool wrappers")
    finally:
        mcp_runtime.stdio_client = original_stdio_client
        mcp_runtime.ClientSession = original_client_session
        clear()

    print("\n🎉 All MCP runtime tests passed")


asyncio.run(main())