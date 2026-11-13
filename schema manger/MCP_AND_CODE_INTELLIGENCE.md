# MCP And Code Intelligence

Updated: 2026-11-13.

## Design

The runtime remains the host: `query.py`, `execute.py`, permissions, Task DAGs, budgets, and persistence stay authoritative. The official MCP Python SDK supplies the stdio client protocol; remote MCP tools are adapted to the local `Tool` contract. No agent framework replaces the existing planner or query loop.

`ToolSearch` is the always-loaded discovery entry point. It first ranks configured server names and descriptions, then connects only the best-matching one or two stdio servers. After interactive approval, it discovers their descriptors and activates at most five matching schemas in the current query context. Repeated searches replace the active set. Each external tool call independently requires interactive approval, including in bypass-permissions mode. MCP metadata and results are untrusted data. MCP tools are not granted to Task workers or the Phase 0 runner.

## Configuration

By default, the manager reads `~/.agent/mcp_servers.json`. `AGENT_MCP_CONFIG` can point to another user-controlled file. The file must contain an explicit `servers` object; project files are not used to launch processes.

```json
{
  "servers": {
    "tracker": {
      "description": "Issue tracking and pull requests.",
      "command": "uvx",
      "args": ["trusted-tracker-mcp-server"],
      "env": {
        "TRACKER_TOKEN": "${TRACKER_TOKEN}"
      }
    }
  }
}
```

Only a small allowlist of non-secret process environment values is inherited. Other variables must be explicitly mapped. A value exactly matching `${NAME}` is resolved from the parent environment; missing variables fail closed. Commands and arguments are passed directly to the subprocess API without a shell. `cwd`, when provided, must be an existing absolute directory.

## Current Limits

- The optional code-intelligence environment is installed from `requirements-code-intel.txt` and `package.json` in this folder. Tree-Sitter symbol extraction currently covers Python, TypeScript/TSX, JavaScript, Go, and Rust; the grammar pack's broader language count is not the same as extraction coverage.
- From this directory, install with `python -m pip install -r requirements-code-intel.txt` and `npm install`. `AstGrep` patterns support Python, TypeScript, TSX, JavaScript, Go, and Rust in this runtime.
- `IndexCode` creates a private mode-0600 SQLite index under `~/.agent/code-index` (override with `AGENT_CODE_INDEX_DIR`), incrementally reuses content hashes, and reports syntax errors/skipped files. Task-scoped indexing is in-memory and contains only manifest-visible files.
- `FindCallers`/`FindCallees` are syntactic name-match candidates; `FindImports` reports syntax-level import strings. They are not a resolved graph and must never be used as proof that code is unused.
- `AstGrep` uses a pinned local CLI and is read-only. Pattern search is bounded to explicitly enumerated files; rewriting is not exposed.
- Pyright references/definitions run on demand over one project root or a Task-scope snapshot. Multi-root/worktree correctness, diagnostics, rename, and cache reuse are not implemented yet.
- Stdio transport only. Streamable HTTP, OAuth, remote server auth, and gateways are not implemented.
- Connecting starts configured subprocesses as the current OS user. Per-tool permission prompts are not an OS sandbox; select and configure trusted servers.
- Server tool schemas are checked locally, remote `$ref` resolution is rejected, and schemas/results are bounded. Descriptor hashes are recorded for diagnostics but there is not yet a persistent approval manifest or `tools/list_changed` refresh flow.
- Tool discovery uses bounded local keyword matching, not BM25 or a third-party gateway. Only selected schemas are sent to the model.
- MCP resources, prompts, sampling, subscriptions, and MCP Apps are not integrated.
- The current coordinator deliberately does not pass MCP capabilities to child workers.

## Existing Code-Intelligence Layer

`tools/search.py` provides bounded literal `Grep` and filename `Glob` over the project, excluding common generated and credential-like paths. Task-worker searches are restricted to each worker's declared file manifest. `tools/code_intelligence.py` provides Python `GetEditContext`, optional Jedi references, tested Pyright references/definition, Tree-Sitter symbols/imports/static call candidates and a token-bounded map, plus bounded ast-grep structural search. Persistent indexes are parent-only; Task queries use manifest-limited in-memory indexes/snapshots. All file outputs distinguish syntax candidates from semantic results. `Edit` retains the read-before-write and stale-file checks. `loop_guard.py` warns on repeated actions, and warnings now reach the model before a hard block.

The headless index and map are not the Aider RepoMap implementation or a complete resolved call graph. Pyright is a real LSP-backed Python navigation layer, but there is still no semantic rename or diagnostics UI. Jedi remains an optional fallback. The existing top-level `aider_mcp_server.py` is not wired into this runtime and its auto-commit behavior is not used here.

## Next Evaluation

1. Run `test_code_index.py`, `test_code_intelligence.py`, `test_pyright_lsp.py`, `test_mcp_runtime.py`, `test_search.py`, `test_execute.py`, Task contract/query integration tests, and then the full offline suite.
2. Exercise one explicitly trusted local stdio server with a human reviewing startup and each call. Do not use a live remote service or credentials in an offline test.
3. Measure context size, reference recall, and edit correctness for AST/Jedi, Pyright, and the static index. Keep unknown/dynamic references explicit; do not make safe-delete decisions from a static candidate graph.
4. Validate Pyright on real multi-root and detached-worktree cases before adding persistent LSP session reuse or diagnostics.
5. Keep embedding search and large agent frameworks optional. They overlap with existing search, workflow, and orchestration layers and should be justified by measured task gaps.

No performance, security, language-coverage, or upstream-maintenance claims from pasted research are treated as verified local evidence.
