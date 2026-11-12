# Integrations and memory

Edit the **active instance configuration**, then restart the application to reconnect configured services. For a new home, that file is the one printed by `fresh`; see [running and recovery](running.md). Settings → Connections exposes the current connection controls and Secret Store.

## Memory

The home platform stores canonical, actor-scoped memory and conversations in SQLite. It works without ChromaDB, Elasticsearch, embeddings, or an external memory service. Default recall uses lexical matching. Passive high-confidence conversation-window injection requires a supporting optional index; the default lexical adapter does not provide semantic recall.

To add an index, install the relevant extra in the environment that runs TamAGI:

```bash
.venv/bin/python -m pip install -e '.[chroma]'
# Or:
.venv/bin/python -m pip install -e '.[elasticsearch]'
```

Select `memory.backend` and its connection/settings, then set `runtime.index_projection: true`. Installing a package alone does not enable projection. In a supervised installation, retain dependency changes through the release workflow instead of patching a selected release’s venv.

The startup adapter schedules an index rebuild when projection is available. `POST /api/runtime/memory/reindex` requests a rebuild explicitly. Failures are reported as projection errors while source memories remain in SQLite. These indexes do not replace backups. Restored instances need the optional dependencies and endpoint reconnected before indexing can work.

The Elasticsearch adapter supports semantic/lexical fusion and recency ranking using caller-generated embeddings and ordinary search primitives. Its conversation recall can inject one selected three-message window and expose further candidates through `expand_conversation_memory`. Read [memory.context](../config.example.yaml) for limits and thresholds. Changing the embedding model requires rebuilding the index consistently.

Non-platform installations instead use the configured legacy memory backend as their memory store. ChromaDB has a keyword-only in-memory fallback when its dependency is unavailable; do not treat that fallback as durable vector storage.

## Web search

```yaml
web_search:
  provider: "duckduckgo"
```

DuckDuckGo needs no API key, but still needs network access and can fail or be rate-limited. Alternatives:

| Provider | Configuration |
| --- | --- |
| `brave` | `web_search.brave_api_key`, or the loader’s `TAMAGI_BRAVE_API_KEY` override. |
| `searxng` | `web_search.searxng_url` pointing at your reachable instance. |

The MCP Secret Store’s `{secret: NAME}` syntax supplies stdio server environment values. It does not automatically resolve the scalar `llm.api_key` or `web_search.brave_api_key` settings. Use supported environment overrides if you want to keep those out of YAML.

## MCP tools

TamAGI discovers configured MCP tools at startup and registers them in its skill registry. The adapter supports **stdio** subprocesses and the older **HTTP+SSE** transport. Native Streamable HTTP transport is not configured by a `transport: streamablehttp` option in this version; use a separately installed compatible stdio bridge when needed.

For a local server you have installed:

```yaml
mcp:
  servers:
    - name: local-tools
      transport: stdio
      command: "/absolute/path/to/your-mcp-server"
      args: []
      env:
        SERVICE_TOKEN:
          secret: SERVICE_TOKEN
      include_tools:
        - lookup
```

For an HTTP+SSE server:

```yaml
mcp:
  servers:
    - name: remote-tools
      transport: sse
      url: "http://your-server:8080/sse"
```

For stdio, resolved `env` values are added to the child process environment when the connection is created, not on every tool call. **The current SSE connector does not forward those values as authentication headers**, despite an older source comment suggesting otherwise. The SSE example above therefore assumes an endpoint that needs no such headers. Use a compatible, separately configured stdio bridge for authenticated remote services; adding `Authorization` under `env` does not authenticate the native SSE connection.

A nonempty `include_tools` list takes precedence over `exclude_tools`; otherwise the exclusion list filters discovered names. These are tool names returned by the server. The current registry uses bare names, so choose integrations without conflicting built-in or other-server names.

The platform checks external-action delegation at dispatch. MCP tools that do not advertise `readOnlyHint` require operator delegation for their `mcp:<tool_name>` scope. Review and grant or revoke configured actions in **Settings**. A server’s read-only annotation is a declaration used by the adapter, not independent proof that it has no external effect.

Use absolute executable paths in service configurations. A command installed in a user’s `~/.local/bin` or activated shell may not be on the service’s PATH. The server package, bridge, credentials, endpoint, and its permissions are separate installation requirements.

`GET /api/mcp/status` reports configured server names, transport, connection state, and registered tools. The configured connection remains open for the application lifetime. Tool failures remain failures even when a server returns accompanying text.

## Credential storage

The Secret Store attempts an OS keyring, with a Fernet-encrypted file fallback under the active instance’s `data/` directory. The fallback key and encrypted data are local files with restrictive permissions; anyone who can read both can decrypt them. Host root retains that authority.

Use **Settings → Connections → Secret Store** to add credentials without placing their values in an example configuration. The API supports:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/secrets` | List file-backed names; OS-keyring-only names are not enumerable. |
| `POST /api/secrets/{name}` | Set a value using JSON `{"value": "..."}`. |
| `DELETE /api/secrets/{name}` | Delete by name. |

Authenticated sessions are required when login is enabled, including all home-platform instances. API scripts need a valid session cookie; mutating requests that supply an Origin must use the application’s origin. The unauthenticated `curl` snippets in earlier documentation are not sufficient for a home-platform installation.

The API does not return stored credential values, and the adapter does not deliberately inject its environment configuration into model prompts. A tool result, shell output, or authored file can nevertheless expose a secret. Avoid claims that a model can never see one. Portable backups exclude dedicated credential stores and sessions; reconnect secrets after restoration.

## Generated tools

For local generated capabilities, use the [subprocess-tool lifecycle](workspace-layout.md#building-a-callable-tool): propose, test, evaluate, and activate. Dropping a Python class in a folder does not register a home-platform tool. Comparative worker-protocol checks establish availability; task-specific usefulness needs corresponding runner-owned scenarios.
