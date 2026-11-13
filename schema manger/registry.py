"""registry.py — the tool lookup table. A dict, not a filter function."""
from tool import Tool

TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Register a tool. Returns it so you can chain: bash = register(build_tool(...))"""
    TOOLS[tool.name] = tool
    return tool


def unregister(name: str, expected: Tool | None = None) -> None:
    """Remove a tool, optionally only when it is still the expected instance."""
    current = TOOLS.get(name)
    if current is not None and (expected is None or current is expected):
        TOOLS.pop(name, None)


def find(name: str) -> Tool | None:
    return TOOLS.get(name)


def all_tools() -> list[Tool]:
    return list(TOOLS.values())


def clear() -> None:
    """Clear the registry — useful for tests."""
    TOOLS.clear()