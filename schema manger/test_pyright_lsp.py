"""Offline integration test for the installed Pyright stdio language server."""
import asyncio
import shutil
import tempfile
from pathlib import Path

from lsp_client import JsonRpcStdioClient


async def main() -> None:
    command = shutil.which("pyright-langserver")
    if not command:
        print("Pyright language server unavailable; skipped integration test")
        return

    with tempfile.TemporaryDirectory() as root:
        source = (
            "def calculate(value: int) -> int:\n"
            "    return value + 1\n\n"
            "def run() -> int:\n"
            "    return calculate(41)\n"
        )
        path = Path(root) / "module.py"
        path.write_text(source, encoding="utf-8")
        client = await JsonRpcStdioClient.start(root, command=command)
        try:
            await client.open_document(str(path), source)
            references = await client.references(str(path), line=0, character=4)
            lines = sorted({
                int(location["range"]["start"]["line"]) + 1
                for location in references
            })
            assert lines == [1, 5], (lines, references)
            print("✅ Pyright LSP initializes and resolves Python references")

            definition = await client.definition(str(path), line=4, character=11)
            locations = definition if isinstance(definition, list) else [definition]
            assert any(
                location
                and int(location["range"]["start"]["line"]) == 0
                for location in locations
            ), definition
            print("✅ Pyright LSP resolves Python definitions")
        finally:
            await client.close()

    print("\n🎉 Pyright LSP tests passed")


asyncio.run(main())
