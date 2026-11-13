"""Minimal bounded JSON-RPC client for Pyright's stdio Language Server."""
from __future__ import annotations

import asyncio
import json
from typing import Any


MAX_LSP_MESSAGE_BYTES = 8 * 1024 * 1024
LSP_START_TIMEOUT_SECONDS = 20
LSP_REQUEST_TIMEOUT_SECONDS = 30


class LspProtocolError(RuntimeError):
    pass


class JsonRpcStdioClient:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()

    @classmethod
    async def start(cls, root: str, command: str = "pyright-langserver") -> "JsonRpcStdioClient":
        process = await asyncio.create_subprocess_exec(
            command,
            "--stdio",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        client = cls(process)
        try:
            await asyncio.wait_for(
                client.request("initialize", {
                    "processId": None,
                    "clientInfo": {"name": "schema-manager", "version": "1"},
                    "rootUri": _path_to_uri(root),
                    "workspaceFolders": [{"uri": _path_to_uri(root), "name": "project"}],
                    "capabilities": {
                        "workspace": {
                            "configuration": True,
                            "workspaceFolders": True,
                        },
                        "textDocument": {
                            "definition": {"dynamicRegistration": False},
                            "references": {"dynamicRegistration": False},
                        },
                    },
                    "initializationOptions": {
                        "python": {"pythonPath": __import__("sys").executable},
                    },
                }),
                timeout=LSP_START_TIMEOUT_SECONDS,
            )
            await client.notify("initialized", {})
            return client
        except BaseException:
            await client.close()
            raise

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.process.returncode is not None or self.process.stdin is None:
            raise LspProtocolError("Pyright language server is not running")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        async with self._write_lock:
            self.process.stdin.write(header + body)
            await self.process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise LspProtocolError("Pyright language server has no stdout stream")
        header = await self.process.stdout.readuntil(b"\r\n\r\n")
        content_length = None
        for line in header.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                content_length = int(line.split(b":", 1)[1].strip())
                break
        if content_length is None or content_length < 0 or content_length > MAX_LSP_MESSAGE_BYTES:
            raise LspProtocolError("Pyright returned an invalid or oversized LSP message")
        body = await self.process.stdout.readexactly(content_length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise LspProtocolError("Pyright returned a non-object JSON-RPC message")
        return payload

    async def _answer_server_request(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            return
        if method == "workspace/configuration":
            items = message.get("params", {}).get("items", [])
            result: Any = [{} for _item in items]
        else:
            result = None
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        async with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            await self._send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            })
            while True:
                message = await asyncio.wait_for(
                    self._read_message(),
                    timeout=LSP_REQUEST_TIMEOUT_SECONDS,
                )
                if "method" in message:
                    await self._answer_server_request(message)
                    continue
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message["error"]
                    raise LspProtocolError(
                        f"Pyright {method} failed ({error.get('code')}): "
                        f"{str(error.get('message', 'unknown error'))[:500]}"
                    )
                return message.get("result")

    async def open_document(self, path: str, source: str) -> None:
        await self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": _path_to_uri(path),
                "languageId": "python",
                "version": 1,
                "text": source,
            },
        })

    async def references(
        self,
        path: str,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> list[dict[str, Any]]:
        result = await self.request("textDocument/references", {
            "textDocument": {"uri": _path_to_uri(path)},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_declaration},
        })
        return result if isinstance(result, list) else []

    async def definition(self, path: str, line: int, character: int) -> Any:
        return await self.request("textDocument/definition", {
            "textDocument": {"uri": _path_to_uri(path)},
            "position": {"line": line, "character": character},
        })

    async def close(self) -> None:
        if self.process.returncode is not None:
            return
        try:
            await asyncio.wait_for(
                self.request("shutdown", {}),
                timeout=3,
            )
            await self.notify("exit", {})
        except Exception:
            pass
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()


def _path_to_uri(path: str) -> str:
    from pathlib import Path
    return Path(path).resolve().as_uri()


__all__ = ["JsonRpcStdioClient", "LspProtocolError"]
