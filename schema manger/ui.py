"""Textual terminal UI for the schema manager agent runtime."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Collapsible, Footer, Header, Input, Label, LoadingIndicator,
    Markdown, Static,
)

from api import get_active_model, list_models, set_active_model
from main import _persist_query_messages, build_context, build_full_system_prompt
from messages import (
    AssistantMessage, ProgressMessage, ToolResultBlock, ToolResultMessage,
    ToolUseBlock, UserMessage as AgentUserMessage, create_assistant_message,
    create_tool_result, create_user_message,
)
from query import QueryParams, query
from query_transitions import Terminal
from registry import all_tools
from state import FileStateCache


class AbortController:
    """Small awaitable abort signal shared by query and running tools."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None

    @property
    def signal(self) -> AbortController:
        return self

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self, reason: str = "interrupt") -> None:
        if self.aborted:
            return
        self.reason = reason
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


class StreamingMarkdown(Markdown):
    """Throttle Markdown reconstruction while streamed text arrives."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._buffer = ""
        self._last_render = 0.0

    def append_text(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        now = time.monotonic()
        if "\n" in text or now - self._last_render >= 0.05:
            self._render_buffer()

    def finish(self) -> None:
        self._render_buffer()

    def _render_buffer(self) -> None:
        self._last_render = time.monotonic()
        self.update(self._buffer)


class ToolCard(Collapsible):
    """One tool call, updated in place from running to its final result."""

    def __init__(self, tool_use_id: str, tool_name: str, summary: str) -> None:
        self.tool_use_id = tool_use_id
        self.tool_name = tool_name
        self._summary = summary
        self._progress_lines: list[str] = []
        self._body = Static("Waiting for result", id="tool-body", markup=False)
        super().__init__(
            self._body,
            title=f"RUNNING  {tool_name}({summary})",
            collapsed=True,
        )

    def append_progress(self, text: str) -> None:
        if not text:
            return
        self._progress_lines.append(text.rstrip())
        self._progress_lines = self._progress_lines[-8:]
        self._body.update("\n".join(self._progress_lines))

    def set_done(self, result_text: str, is_error: bool = False) -> None:
        state = "ERROR" if is_error else "DONE"
        self.title = f"{state}  {self.tool_name}({self._summary})"
        lines = (result_text or "(no output)").strip().splitlines()
        preview = "\n".join(lines[:16])
        if len(lines) > 16:
            preview += f"\n... {len(lines) - 16} more lines"
        self._body.update(preview)
        self.collapsed = not is_error


class UserMessage(Markdown):
    def __init__(self, text: str) -> None:
        super().__init__(f"**YOU**\n\n{text}")


class SystemNotice(Static):
    def __init__(self, text: str, level: str = "info") -> None:
        super().__init__(markup=False)
        self._text = text
        self._level = level.upper()

    async def on_mount(self) -> None:
        self.update(f"{self._level}  {self._text}")


class PlanPanel(Collapsible):
    def __init__(self) -> None:
        super().__init__(
            Markdown(""), title="PLAN", collapsed=True, id="plan-panel",
        )
        self.border_title = "SESSION PLAN"

    def refresh_from_plan(self, plan: dict | None) -> None:
        markdown = self.query_one(Markdown)
        items = plan.get("items", []) if plan else []
        self.display = bool(items)
        if not items:
            markdown.update("")
            self.title = "PLAN"
            return
        lines = [f"**{plan['objective']}**", ""]
        for item in items:
            marker = {"completed": "x", "in_progress": ">", "pending": " "}.get(
                item.get("status"), " ",
            )
            lines.append(f"- [{marker}] {item['content']}")
        completed = sum(item.get("status") == "completed" for item in items)
        lines.append(f"\n{completed}/{len(items)} complete")
        if plan.get("focus"):
            lines.append(f"\nFocus: {plan['focus']}")
        self.title = f"PLAN  {completed}/{len(items)}"
        markdown.update("\n".join(lines))


class StatusBar(Horizontal):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.activity = "Ready"

    def compose(self) -> ComposeResult:
        yield LoadingIndicator(id="spinner")
        yield Label("Ready", id="activity")
        yield Label("0 tokens", id="tokens")

    def set_running(self, activity: str) -> None:
        self.activity = activity
        self.query_one("#spinner", LoadingIndicator).display = True
        self.query_one("#activity", Label).update(activity)

    def set_idle(self) -> None:
        self.activity = "Ready"
        self.query_one("#spinner", LoadingIndicator).display = False
        self.query_one("#activity", Label).update("Ready")

    def set_tokens(self, count: int) -> None:
        self.query_one("#tokens", Label).update(f"{count:,} tokens")


class PermissionScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny", show=False),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-dialog"):
            yield Label("PERMISSION REQUIRED", id="permission-title")
            yield Static(self._message, id="permission-message", markup=False)
            with Horizontal(id="permission-actions"):
                yield Button("Allow", id="allow", variant="success")
                yield Button("Deny", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class AgentApp(App):
    TITLE = "SCHEMA MANAGER"

    CSS = """
    Screen { layout: vertical; background: $background; }
    Header { dock: top; }
    #workspace { height: 1fr; min-height: 8; }
    #session-sidebar { width: 30; min-width: 24; padding: 1; border-right: tall $primary; }
    #sidebar-title { height: 1; margin-bottom: 1; text-style: bold; color: $accent; }
    #new-session-button, #tools-button { width: 1fr; height: 3; margin-bottom: 1; }
    #session-filter { margin-bottom: 1; }
    #session-list { height: 1fr; min-height: 5; }
    .session-button { width: 1fr; height: 3; margin-bottom: 1; content-align: left middle; }
    #sidebar-status { height: auto; max-height: 5; color: $text-muted; margin-top: 1; }
    #main-pane { width: 1fr; }
    #session-heading { height: 2; padding: 0 2; background: $panel; content-align: left middle; }
    #transcript { height: 1fr; min-height: 8; padding: 0 2; }
    UserMessage { margin: 1 0 0 0; padding: 0 1; border-left: thick $accent; }
    StreamingMarkdown { margin: 0 0 1 2; padding: 0 1; }
    ToolCard { margin: 0 0 1 2; }
    SystemNotice { margin: 0 0 1 2; color: $text-muted; }
    #plan-panel { max-height: 8; margin: 0 2; }
    #status { height: 1; padding: 0 2; background: $panel; margin: 0 2; }
    #spinner { width: 3; }
    #activity { width: 1fr; }
    #tokens { width: 16; text-align: right; }
    #composer { height: 3; margin: 0 2 1 2; border: tall $accent; }
    #tools-view { height: 1fr; padding: 1 2; }
    #permission-dialog { width: 72; height: auto; padding: 1 2; border: thick $warning; background: $surface; }
    #permission-title { text-style: bold; color: $warning; margin-bottom: 1; }
    #permission-message { max-height: 12; margin-bottom: 1; }
    #permission-actions { height: 3; align-horizontal: right; }
    #permission-actions Button { margin-left: 1; }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Exit", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "new_session", "New session"),
        Binding("ctrl+1", "show_current", "Current"),
        Binding("ctrl+2", "show_tools", "Tools"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.ctx = build_context()
        self.ctx.abort_controller = AbortController()
        self._worker = None
        self._current_markdown: StreamingMarkdown | None = None
        self._tool_cards: dict[str, ToolCard] = {}
        self._turn_tokens = 0
        self._exit_when_idle = False
        self._permission_lock = asyncio.Lock()
        self._permission_waiters = 0
        self._session_buttons: dict[str, str] = {}
        self._selected_session_id = self.ctx.session_id
        self._active_view = "current"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            with Vertical(id="session-sidebar"):
                yield Label("RECENT SESSIONS", id="sidebar-title")
                yield Button("+ New session", id="new-session-button", variant="primary")
                yield Button("Tool catalog", id="tools-button")
                yield Input(placeholder="Filter sessions", id="session-filter")
                with VerticalScroll(id="session-list"):
                    pass
                yield Static("", id="sidebar-status", markup=False)
            with Vertical(id="main-pane"):
                yield Label("CURRENT SESSION", id="session-heading")
                with VerticalScroll(id="transcript"):
                    pass
                with VerticalScroll(id="tools-view"):
                    yield Static("", id="tool-catalog", markup=False)
                yield PlanPanel()
                yield StatusBar(id="status")
                yield Input(placeholder="Message the agent...  (/help for commands)", id="composer")
        yield Footer()

    async def on_mount(self) -> None:
        self._update_header()
        self.query_one(StatusBar).set_idle()
        self.query_one("#composer", Input).focus()
        self._append_widget(SystemNotice(
            f"Ready on {get_active_model()} | {len(all_tools())} tools",
        ))
        self.query_one("#tools-view").display = False
        await self._refresh_sessions()
        self._refresh_sidebar_status()
        self._refresh_plan()

    @on(Input.Changed, "#session-filter")
    async def on_session_filter_changed(self, event: Input.Changed) -> None:
        await self._refresh_sessions(event.value)

    @on(Button.Pressed, "#new-session-button")
    async def on_new_session_button(self, _event: Button.Pressed) -> None:
        await self.action_new_session()

    @on(Button.Pressed, "#tools-button")
    def on_tools_button(self, _event: Button.Pressed) -> None:
        self.action_show_tools()

    @on(Button.Pressed, ".session-button")
    async def on_session_button(self, event: Button.Pressed) -> None:
        session_id = self._session_buttons.get(event.button.id or "")
        if session_id:
            await self._select_session(session_id)

    @on(Input.Submitted, "#composer")
    def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if self._worker is not None and not self._worker.is_finished:
            self._append_widget(SystemNotice("A turn is already running.", "warn"))
            return
        if self._handle_command(text):
            return
        self._append_widget(UserMessage(text))
        self._start_turn(text)

    def _handle_command(self, text: str) -> bool:
        if text in ("/quit", "/exit", "/q"):
            self.exit()
        elif text == "/help":
            self._append_widget(SystemNotice(
                "/tools lists capabilities | /model [id] switches provider model | "
                "/clear starts a fresh session | Ctrl+C exits",
            ))
        elif text == "/tools":
            tools = "\n".join(
                f"{tool.name}: {tool.search_hint or tool.description}"
                for tool in all_tools()
            )
            self._append_widget(SystemNotice(tools))
        elif text == "/model":
            available = ", ".join(list_models()) or "No configured provider keys"
            self._append_widget(SystemNotice(
                f"Current: {get_active_model()} | Available: {available}",
            ))
        elif text.startswith("/model "):
            model = text[7:].strip()
            try:
                set_active_model(model)
                self._update_header()
                self._append_widget(SystemNotice(f"Model set to {model}"))
            except Exception as error:
                self._append_widget(SystemNotice(str(error), "error"))
        elif text in ("/clear", "/new"):
            self.run_worker(self.action_new_session(), group="session")
        else:
            return False
        return True

    def _start_turn(self, text: str) -> None:
        self.ctx.abort_controller = AbortController()
        self._turn_tokens = 0
        self._current_markdown = None
        self._tool_cards.clear()
        composer = self.query_one("#composer", Input)
        composer.disabled = True
        self.query_one(StatusBar).set_tokens(0)
        self.query_one(StatusBar).set_running("Thinking")
        self._worker = self.run_turn_worker(text)

    @work(exclusive=True, group="turn")
    async def run_turn_worker(self, text: str) -> None:
        state = self.ctx.get_app_state()
        state.messages.append(create_user_message(text))
        params = QueryParams(
            messages=state.messages,
            system_prompt=build_full_system_prompt(),
            tool_use_context=self.ctx,
            can_use_tool=self._can_use_tool,
            query_source="repl_main_thread",
            get_turn_output_tokens=lambda: self._turn_tokens,
        )
        try:
            async for event in query(params):
                await self._dispatch_event(event)
        except Exception as error:
            self._append_widget(SystemNotice(
                f"{type(error).__name__}: {error}", "error",
            ))
        finally:
            if self._current_markdown is not None:
                self._current_markdown.finish()
            _persist_query_messages(self.ctx, state.messages)
            self._refresh_plan()
            await self._refresh_sessions()
            self._refresh_sidebar_status()
            self.query_one(StatusBar).set_idle()
            composer = self.query_one("#composer", Input)
            composer.disabled = False
            composer.focus()
            if self._exit_when_idle:
                self.exit()

    async def _dispatch_event(self, event: Any) -> None:
        if isinstance(event, Terminal):
            if event.reason != "completed":
                self._append_widget(SystemNotice(f"Turn ended: {event.reason}", "warn"))
            return
        if isinstance(event, dict):
            if event.get("type") == "stream_request_start":
                self.query_one(StatusBar).set_running("Thinking")
            elif event.get("type") == "runtime_activity":
                self._show_runtime_activity(event)
            elif event.get("type") == "tombstone":
                self._append_widget(SystemNotice("Retrying with configured fallback."))
            return

        event_type = getattr(event, "type", None)
        if event_type == "assistant":
            if getattr(event, "is_api_error", False):
                self._append_widget(SystemNotice(event.content or "Provider error", "error"))
            elif event.content:
                if self._current_markdown is None:
                    self._current_markdown = StreamingMarkdown()
                    self._append_widget(self._current_markdown)
                self._current_markdown.append_text(event.content)
            for call in getattr(event, "tool_calls", None) or []:
                summary = self._summarize_input(call.input)
                card = ToolCard(call.id, call.name, summary)
                self._tool_cards[call.id] = card
                self._current_markdown = None
                self._append_widget(card)
                self.query_one(StatusBar).set_running(f"{call.name}  {summary}")
            usage = getattr(event, "usage", None) or {}
            if usage:
                self._turn_tokens = sum(int(usage.get(key, 0) or 0) for key in (
                    "input_tokens", "output_tokens", "cache_read_input_tokens",
                ))
                self.query_one(StatusBar).set_tokens(self._turn_tokens)
        elif event_type == "tool_result" or isinstance(event, ToolResultMessage):
            tool_use_id = getattr(event, "tool_use_id", None)
            card = self._tool_cards.pop(tool_use_id, None)
            if card is not None:
                card.set_done(
                    str(getattr(event, "content", "")),
                    is_error=bool(getattr(event, "is_error", False)),
                )
            self._refresh_plan()
        elif event_type == "progress" or isinstance(event, ProgressMessage):
            data = getattr(event, "data", None) or {}
            if data.get("type") == "task_lifecycle":
                task_id = data.get("task_id", "task")
                status = data.get("status", "updated")
                dependencies = data.get("needs") or []
                detail = f" ({', '.join(dependencies)})" if dependencies else ""
                duration = data.get("duration_ms")
                if duration is not None:
                    detail += f" · {duration} ms"
                error_type = data.get("error_type")
                if error_type:
                    detail += f" · {error_type}"
                level = "error" if status == "failed" else "warn" if status == "blocked" else "info"
                self._append_widget(SystemNotice(
                    f"Task {task_id}: {status}{detail}", level,
                ))
            elif data.get("type") in ("plan_activity", "memory_activity"):
                self._show_runtime_activity(data)
            elif data.get("type") == "bash_progress":
                tool_id = getattr(event, "tool_use_id", "")
                card = self._tool_cards.get(tool_id)
                output = data.get("output", "")
                if card is not None:
                    card.append_progress(output)
                self.query_one(StatusBar).set_running(
                    f"Bash  {str(output).strip()[-72:]}" or "Bash running",
                )
        elif event_type == "system":
            self._append_widget(SystemNotice(getattr(event, "content", "")))

    async def _can_use_tool(self, decision: Any) -> bool:
        self._permission_waiters += 1
        try:
            async with self._permission_lock:
                self.query_one(StatusBar).set_running("Waiting for permission")
                return bool(await self.push_screen_wait(
                    PermissionScreen(getattr(decision, "message", str(decision))),
                ))
        finally:
            self._permission_waiters -= 1
            if self._permission_waiters == 0 and self.is_running:
                worker = self._worker
                if worker is not None and not worker.is_finished:
                    self.query_one(StatusBar).set_running("Working")
                else:
                    self.query_one(StatusBar).set_idle()

    def _refresh_plan(self) -> None:
        store = getattr(self.ctx, "_store", None)
        session_id = getattr(self.ctx, "session_id", None)
        plan = store.get_plan(session_id) if store is not None and session_id else None
        self.query_one(PlanPanel).refresh_from_plan(plan)

    async def _refresh_sessions(self, filter_text: str = "") -> None:
        store = getattr(self.ctx, "_store", None)
        if store is None:
            return
        session_list = self.query_one("#session-list", VerticalScroll)
        await session_list.remove_children()
        self._session_buttons.clear()

        needle = filter_text.strip().casefold()
        sessions = store.list_sessions(limit=100)
        visible = []
        for session in sessions:
            title = " ".join(str(session.get("title") or "New session").split())
            last_message = " ".join(str(session.get("last_message") or "").split())
            if needle and needle not in title.casefold() and needle not in last_message.casefold():
                continue
            visible.append((session, title))

        if not visible:
            session_list.mount(Static("No matching sessions", markup=False))
            return

        for index, (session, title) in enumerate(visible):
            session_id = str(session["session_id"])
            prefix = "CURRENT  " if session_id == self._selected_session_id else ""
            title = title if len(title) <= 23 else title[:20] + "..."
            updated = datetime.fromtimestamp(
                float(session.get("updated_at") or 0),
            ).strftime("%m-%d %H:%M")
            label = f"{prefix}{title}\n{updated} · {session['message_count']} messages"
            button_id = f"session-item-{index}"
            self._session_buttons[button_id] = session_id
            session_list.mount(Button(label, id=button_id, classes="session-button"))

    def _refresh_sidebar_status(self) -> None:
        manager = getattr(self.ctx, "mcp_manager", None)
        configured = bool(
            manager is not None
            and getattr(manager, "config_path", None)
            and manager.config_path.is_file()
        )
        connected = len(getattr(manager, "_connections", {})) if manager else 0
        mcp_status = (
            f"MCP: {connected} connected"
            if connected else "MCP: configured" if configured else "MCP: not configured"
        )
        tools = {tool.name for tool in all_tools()}
        code_features = [
            name for name in ("Grep", "Glob", "RepoMap", "FindReferences", "PyrightReferences", "AstGrep")
            if name in tools
        ]
        self.query_one("#sidebar-status", Static).update(
            mcp_status + "\nCode: " + ", ".join(code_features),
        )

    async def _select_session(self, session_id: str) -> None:
        if self._worker is not None and not self._worker.is_finished:
            self._append_widget(SystemNotice(
                "Interrupt the active turn before switching sessions.", "warn",
            ))
            return
        store = getattr(self.ctx, "_store", None)
        if store is None or session_id == self._selected_session_id:
            return

        rows = store.get_session_messages(session_id, limit=200)
        restored = []
        for row in rows:
            role = row.get("role")
            if role == "user":
                restored.append(create_user_message(row.get("content") or ""))
            elif role == "assistant":
                calls = [
                    ToolUseBlock(
                        id=str(call.get("id", "")),
                        name=str(call.get("name", "")),
                        input=call.get("input") or {},
                    )
                    for call in row.get("tool_calls") or []
                ]
                restored.append(create_assistant_message(
                    content=row.get("content") or None,
                    tool_calls=calls,
                    stop_reason="tool_calls" if calls else "end_turn",
                ))
            elif role == "tool":
                restored.append(create_tool_result(
                    str(row.get("tool_use_id") or ""),
                    str(row.get("content") or ""),
                ))

        pending_calls = {
            call.id: call
            for message in restored
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        }
        completed_calls = {
            message.tool_use_id for message in restored
            if isinstance(message, ToolResultMessage)
        }
        for tool_id in pending_calls.keys() - completed_calls:
            restored.append(create_tool_result(
                tool_id,
                "Historical tool call was incomplete and was not re-executed.",
                is_error=True,
            ))

        state = self.ctx.get_app_state()
        state.messages[:] = restored
        self.ctx.messages = state.messages
        self.ctx.session_id = session_id
        self._selected_session_id = session_id
        self.ctx.active_mcp_tool_names = set()
        self.ctx._stored_message_ids = [row["id"] for row in rows]
        self.ctx._stored_message_uuids = {message.uuid for message in restored}
        self.ctx._summary_source_offset = len(self.ctx._stored_message_ids)

        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.remove_children()
        self._tool_cards.clear()
        for message in restored:
            if isinstance(message, AgentUserMessage):
                text = message.content if isinstance(message.content, str) else str(message.content)
                transcript.mount(UserMessage(text))
            elif isinstance(message, AssistantMessage):
                if message.content:
                    markdown = StreamingMarkdown()
                    markdown.append_text(message.content)
                    markdown.finish()
                    transcript.mount(markdown)
                for call in message.tool_calls:
                    card = ToolCard(call.id, call.name, self._summarize_input(call.input))
                    self._tool_cards[call.id] = card
                    transcript.mount(card)
            elif isinstance(message, ToolResultMessage):
                card = self._tool_cards.get(message.tool_use_id)
                if card is not None:
                    card.set_done(message.content, is_error=message.is_error)
                else:
                    transcript.mount(SystemNotice(
                        f"Tool result {message.tool_use_id}: {message.content[:160]}",
                        "error" if message.is_error else "info",
                    ))

        session = next(
            (item for item in store.list_sessions(limit=100)
             if item["session_id"] == session_id),
            None,
        )
        title = " ".join(str(session.get("title") or "Session").split()) if session else "Session"
        self.query_one("#session-heading", Label).update(title[:80])
        self._active_view = "current"
        self._show_current_view()
        self._refresh_plan()
        await self._refresh_sessions()
        self._update_header()

    def _show_current_view(self) -> None:
        self._active_view = "current"
        self.query_one("#tools-view").display = False
        self.query_one("#transcript").display = True
        self.query_one(PlanPanel).display = True
        self.query_one(StatusBar).display = True
        self.query_one("#composer", Input).display = True

    def action_show_current(self) -> None:
        self._show_current_view()

    def action_show_tools(self) -> None:
        tools_view = self.query_one("#tools-view", VerticalScroll)
        catalog = self.query_one("#tool-catalog", Static)
        catalog.update("\n\n".join(
            f"{tool.name}\n{tool.search_hint or tool.description}"
            for tool in all_tools()
        ))
        self.query_one("#transcript").display = False
        self.query_one(PlanPanel).display = False
        self.query_one(StatusBar).display = False
        self.query_one("#composer", Input).display = False
        tools_view.display = True
        self._active_view = "tools"

    def _show_runtime_activity(self, event: dict) -> None:
        area = event.get("area")
        if area == "memory":
            source = event.get("source", "memory")
            names = event.get("names") or []
            suffix = f": {', '.join(names)}" if names else ""
            text = f"Memory context injected ({source}, {event.get('count', 0)} items){suffix}"
            level = "info"
        elif event.get("type") == "memory_activity":
            action = event.get("action", "memory")
            status = event.get("status", "updated")
            count = event.get("result_count")
            suffix = f" ({count} results)" if count is not None else ""
            if event.get("node_id") is not None:
                suffix += f" · source {event['node_id']}"
            text = f"Memory {action}: {status}{suffix}"
            level = "error" if status == "failed" else "info"
        elif event.get("type") == "plan_activity":
            if event.get("cleared"):
                text = "Plan completed and cleared"
            else:
                text = (
                    f"Plan updated: {event.get('item_count', 0)} items, "
                    f"{event.get('completed_count', 0)} complete, "
                    f"{event.get('in_progress_count', 0)} active"
                )
            level = "info"
        elif area == "provider":
            status = event.get("status", "updated")
            model = event.get("model", "provider")
            if status == "retrying":
                text = (
                    f"Provider retry: {model} ({event.get('category', 'temporary failure')}, "
                    f"attempt {event.get('attempt', '?')}, "
                    f"wait {event.get('delay_seconds', 0)}s)"
                )
                level = "warn"
            elif status == "fallback":
                text = (
                    f"Provider fallback: {model} → "
                    f"{event.get('alternate_model', 'alternate')} "
                    f"({event.get('category', 'provider failure')})"
                )
                level = "warn"
            else:
                text = f"Provider {status}: {model}"
                level = "error" if status == "failed" else "info"
                if event.get("status_code"):
                    text += f" · HTTP {event['status_code']}"
                if event.get("request_id"):
                    text += f" · request {event['request_id']}"
        elif area == "model":
            status = event.get("status", "updated")
            model = event.get("model", "model")
            text = f"Model {status}: {model}"
            level = "error" if status == "failed" else "info"
            if event.get("status_code"):
                text += f" · HTTP {event['status_code']}"
            if event.get("request_id"):
                text += f" · request {event['request_id']}"
        else:
            return
        self._append_widget(SystemNotice(text, level))

    def _summarize_input(self, inputs: Any) -> str:
        if not isinstance(inputs, dict):
            return ""
        for key in ("file_path", "command", "pattern", "query"):
            if inputs.get(key):
                value = str(inputs[key]).replace("\n", " ")
                return value if len(value) <= 64 else value[:61] + "..."
        return ""

    def _append_widget(self, widget: Any) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        stick_to_bottom = transcript.is_vertical_scroll_end
        transcript.mount(widget)
        if stick_to_bottom:
            self.call_after_refresh(transcript.scroll_end, animate=False)

    def _update_header(self) -> None:
        self.title = f"SCHEMA MANAGER  |  {get_active_model()}"
        self.sub_title = getattr(self.ctx, "session_id", "session")

    def action_interrupt(self) -> None:
        worker = self._worker
        if worker is None or worker.is_finished:
            self.exit()
            return

        controller = getattr(self.ctx, "abort_controller", None)
        if controller:
            controller.abort("interrupt")
        self._exit_when_idle = True
        worker.cancel()
        self.query_one(StatusBar).set_running("Closing")

    async def action_new_session(self) -> None:
        if self._worker is not None and not self._worker.is_finished:
            self._append_widget(SystemNotice(
                "Interrupt the active turn before starting a new session.", "warn",
            ))
            return
        state = self.ctx.get_app_state()
        state.messages.clear()
        state.read_file_cache = FileStateCache()
        self.ctx.read_file_state = state.read_file_cache
        self.ctx.session_id = f"ui-{uuid.uuid4().hex[:12]}"
        self.ctx.active_mcp_tool_names = set()
        self.ctx._stored_message_ids = []
        self.ctx._stored_message_uuids = set()
        self.ctx._summary_source_offset = 0
        await self.query_one("#transcript", VerticalScroll).remove_children()
        await self.query_one("#session-list", VerticalScroll).remove_children()
        self._update_header()
        self.query_one("#session-heading", Label).update("CURRENT SESSION")
        self._show_current_view()
        self._refresh_plan()
        await self._refresh_sessions()
        self._refresh_sidebar_status()
        self._append_widget(SystemNotice("Started a fresh session."))

    async def on_unmount(self) -> None:
        manager = getattr(self.ctx, "mcp_manager", None)
        if manager is not None:
            await manager.close()
        store = getattr(self.ctx, "_store", None)
        if store is not None:
            store.close()


def main() -> None:
    AgentApp().run()


if __name__ == "__main__":
    main()