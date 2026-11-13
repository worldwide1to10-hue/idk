"""Model-facing progressive discovery for configured MCP capabilities."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from permission_types import OtherReason, PermissionAskDecision
from tool import ToolResult, build_tool


class ToolSearchInput(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=160,
        description="Capability or server name to search for, such as 'github issues'.",
    )

    @field_validator("query")
    @classmethod
    def _trim_query(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Tool search query must contain at least two characters")
        return value


async def _search(
    input: ToolSearchInput,
    context: Any,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    manager = getattr(context, "mcp_manager", None)
    if manager is None:
        return ToolResult(data={
            "is_error": True,
            "content": "MCP is not configured in this application context.",
        })
    try:
        result = await manager.search_and_activate(input.query, context)
    except Exception as error:
        import logging
        logging.getLogger(__name__).warning(
            "MCP discovery failed: %s", type(error).__name__,
        )
        return ToolResult(data={
            "is_error": True,
            "content": f"MCP discovery failed: {type(error).__name__}.",
        })
    return ToolResult(data={"is_error": False, "content": result})


async def _ask_for_connection(input: ToolSearchInput, _context: Any):
    return PermissionAskDecision(
        message=(
            "Connect to the user-configured local MCP stdio servers and search "
            f"their tool catalog for {input.query!r}? The server process will run as this user."
        ),
        decision_reason=OtherReason(reason="mcp_server_connection"),
        updated_input=input.model_dump(),
    )


def _map_result(data: dict[str, Any], tool_use_id: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": data.get("content", ""),
        "is_error": bool(data.get("is_error", False)),
    }


ToolSearchTool = build_tool(
    name="ToolSearch",
    description=(
        "Search configured MCP servers and activate up to five matching tool schemas "
        "for this session."
    ),
    prompt=(
        "Use ToolSearch when an external capability may help. It connects only to "
        "user-configured stdio MCP servers after interactive approval and activates "
        "a small matching set of schemas. Each MCP tool call separately requires approval."
    ),
    input_schema=ToolSearchInput,
    call=_search,
    max_result_size_chars=8_000,
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda input, _options: (
        f"ToolSearch({input.get('query', '')})"
    ),
    is_read_only=lambda _input: False,
    is_concurrency_safe=lambda _input: False,
    is_open_world=lambda _input: True,
    check_permissions=_ask_for_connection,
    requires_user_interaction=lambda: True,
)


__all__ = ["ToolSearchInput", "ToolSearchTool"]