# Running and recovering TamAGI

The [README](../README.md#run-a-new-home) is the short path for a new home-platform instance. This guide covers the `codex/dependable-home` branch, which contains the v0.3 runtime during burn-in. The default `main` branch still has the older runtime as of 2026-11-06.

## Environment and configuration

The qualified environment is Linux with Python 3.13 and the pinned [requirements.lock](../requirements.lock). Package metadata permits Python 3.11+, but that is not a claim that the current lock, platform file locking, or recovery tests work on every Python version or OS. Browser access from another device is separate from backend support.

Install Git, Python 3.13 with venv/pip support, and GnuPG for encrypted backups and recovery checks. A model server is a separate dependency; TamAGI expects a compatible chat-completions endpoint with tool-call support. No model weights are bundled.

The fresh-instance command generates a complete configuration at the printed path. Edit that file rather than the repository’s `config.yaml`. The checked-in [config.example.yaml](../config.example.yaml) documents both modes; copying it alone leaves `runtime.enabled: false`.

Common settings:

| Setting | Meaning |
| --- | --- |
| `llm.base_url`, `llm.model`, `llm.api_key` | Endpoint base including `/v1`, exact installed model name, and provider credential. |
| `llm.max_tokens`, `llm.timeout` | Requested reply allowance and request timeout. Choose values supported by your server/model. |
| `llm.num_ctx` | Ollama context option; other servers need their own context configuration. |
| `server.host`, `server.port` | Listen address/port; a fresh home binds to `127.0.0.1`. Use one application worker per instance. |
| `runtime.timezone` | Local scheduling timezone; defaults to `America/New_York`. Change it for your installation. |
| `runtime.index_projection` | Enables an optional retrieval index; defaults to false. SQLite remains the source of memory. |
| `runtime.automatic_deployment` | Defaults to false; activation also requires recovery qualification and verified host-root authority. |

`TAMAGI_CONFIG` selects a configuration for `python -m backend.main`. The `serve --config` command sets it for you. The loader also supports `TAMAGI_LLM_BASE_URL`, `TAMAGI_LLM_MODEL`, `TAMAGI_LLM_API_KEY`, `TAMAGI_PORT`, and `TAMAGI_AUTONOMY_ENABLED`; see [load_config](../backend/config.py) for the complete mapping. Use YAML’s `autonomy.schedule` for timing; the old interval-based settings are obsolete.

A fresh home requires operator login even if an old configuration disabled it. Open `/login` on first start to set the password, then finish onboarding. Autonomous World Thread inference waits until the introduction is complete.

## Access from another device

Use the port chosen at provisioning. For a localhost-bound installation on port 7741, an SSH tunnel can expose it only to your local browser:

```bash
ssh -L 7741:127.0.0.1:7741 user@tamagi-host
```

Then open `http://localhost:7741/login`. For direct LAN access, explicitly configure `server.host` and your network access controls. Browser installation as a PWA and clipboard/service-worker features depend on the browser and a secure context; HTTPS is needed for those features on ordinary remote origins.

The service worker currently caches application assets. It does not deliver push notifications; those belong to the pending household design.

## Scheduling and model allowances

Current autonomy settings are:

```yaml
autonomy:
  enabled: true
  schedule: "*/15 * * * *"
  active_hours_start: 0
  active_hours_end: 24
  resume_after_conversation: 15
runtime:
  timezone: "America/New_York"
```

Merge settings into the generated configuration; do not replace its instance paths. The cron schedule offers an opportunity every fifteen minutes. Active hours are local to `runtime.timezone` in platform mode; `0` through `24` allows all hours. Chosen rest, event waits, pending chat work, onboarding, and error backoff can suppress inference. The post-conversation interval begins from admitted work’s visit deadline, not from closing a browser tab.

Default home-platform allowances:

| Resource | Default |
| --- | --- |
| Concurrent provider calls within one instance | 1 |
| Calls per autonomous activation | 128 |
| Calls per companion activation | 16, drawn from its parent execution |
| Ordinary background calls per rolling day | No cap (`null`); optional positive cap remains enforced |
| Evaluation calls per rolling 24 hours | 32 |
| Calls per user or quiet turn | 20 |
| Explicitly requested quiet continuations | At most 5; at most 120 calls across the chain |

These are ceilings. A one-call reply, refusal, or silent rest can end earlier. Retries and delegated provider calls consume the same applicable allowances. Increasing a limit does not establish better behavior. Separate instances do not yet share provider-time fairness.

Journal displays activity/wake state and World Thread entries. Manual Tick Now bypasses the conversation pause and system-error backoff, while preserving onboarding, chosen sleep/event waits, and rolling allowances. A completed, failed, or interrupted evaluation releases its completion wait; inspect the retained job status to learn which outcome occurred.

The old `interval_minutes`, `inactive_hours_*`, activity weights, Dream Engine endpoints, and per-tool XP explanation are not current home-platform controls.

## Preserve an existing resident

Updating application source and creating a new resident are different operations. `fresh` creates a **separate identity**; it is not an in-place migration. Restoring an encrypted home backup is the supported continuity-preserving recovery path.

Before a legacy-to-home cutover, stop the old process/service and archive it:

```bash
.venv/bin/python -m backend.runtime_cli archive --config /absolute/legacy/config.yaml
```

The command prints the archive directory and completeness status. It preserves source revision, lived files, non-secret configuration, and a logical memory export. An incomplete export blocks fresh cutover. The original is not deleted. Keep this predecessor material separate from the new resident’s hydration and retrieval.

To create an intentionally new home after preservation:

```bash
.venv/bin/python -m backend.runtime_cli fresh \
  --archive /absolute/archive/directory \
  --authority guarded --instance /absolute/new-instance \
  --legacy-config /absolute/legacy/config.yaml --port 7741
```

The instance directory must not already exist. `--legacy-config` reuses only the model transport settings; it does not import identity, history, or integrations. Inspect the generated configuration before using `serve`.

For an existing home-platform resident, retain its complete instance directory and use its same configuration on restart. Do not run two processes against one instance. Use the established release workflow for a supervised installation, and retain an encrypted checkpoint before major changes.

## Authority and supervised releases

`guarded` applies application path/command policies. Run it under an appropriately unprivileged OS account; generated tools and processes are not confined by an OS sandbox.

`unrestricted_host` is intended for an explicitly dedicated Linux host running as actual host UID 0. Container root does not qualify. Root can change the application’s controls, records, and supervisor. External publishing, messaging, purchases, and consequential service actions still require operator authorization unless already delegated.

For an already provisioned dedicated host-root instance, run from its canonical repository (the supplied service assumes `/opt/TamAGI`):

```bash
.venv/bin/python -m backend.runtime_cli qualify --config /absolute/instance/config.yaml
.venv/bin/python -m backend.runtime_cli install-supervisor --config /absolute/instance/config.yaml
sudo install -m 0644 tamagi-platform.service /etc/systemd/system/tamagi-platform.service
sudo systemctl daemon-reload
sudo systemctl enable --now tamagi-platform.service
```

Stop any manually served copy before starting the service. Review [tamagi-platform.service](../tamagi-platform.service) first: it runs as root and checks port **7742**, whereas the README explicitly provisions **7741**. Adjust the unit’s repository paths and `--port` to match your instance. The unit is a dedicated-host template, not the default guarded quick start.

The supervisor runs the exact selected release in `.platform/current` using `.platform/active.json`. Editing the checkout alone does not change the running release. Recovery qualification runs the mechanical suite; it does not complete the behavioral pilot. Qualified automatic deployment can be enabled in Settings. Source compatibility, exact builds, copied-state boot checks, and supervisor recovery remain required.

Useful service commands:

```bash
sudo systemctl status tamagi-platform.service
journalctl -u tamagi-platform.service -f
sudo systemctl stop tamagi-platform.service
```

The older [tamagi.service](../tamagi.service) is a separate legacy template. It uses `/usr/bin/python3`, not the project venv; adjust its `ExecStart` to the installed environment before using it. Keep the legacy and platform services from serving the same resident/port simultaneously.

## Docker and legacy compatibility

The supplied [Compose file](../docker-compose.yml) starts the **legacy** layout: repository `config.yaml`, `data/`, and a workspace mounted at `/workspace`. It does not provision a home-platform instance. For a new legacy installation, copy `config.example.yaml` to a new `config.yaml`, configure the endpoint/authentication and paths, then use:

```bash
docker compose up --build -d
```

Inside a container, `localhost` refers to that container. Use an endpoint address reachable from it; the default host-side Ollama URL is not automatically reachable. Compose publishes port 7741; a plain `python -m backend.main` legacy installation instead uses the configuration’s listen address.

The [Dockerfile](../Dockerfile) uses Python 3.13 and includes the ChromaDB extra by default. `TAMAGI_MEMORY_EXTRA=none` builds core dependencies only; `elasticsearch` selects the alternate legacy extra. For example:

```bash
docker build --build-arg TAMAGI_MEMORY_EXTRA=none -t tamagi:core .
```

That build does not itself enable SQLite platform mode. A containerized home needs a separately provisioned guarded instance, matching persistent paths and configuration, and a container-reachable listen address. The manual Linux quick start avoids the legacy Compose assumptions.

Non-platform operation also retains Python `Skill` directory discovery and legacy memory adapters. Those behaviors should not be used to infer the home platform’s tool, memory, or recovery contracts. The unresolved [graph-save issue](https://github.com/manikmakki/TamAGI/issues/5) still affects legacy graph writes and some platform compatibility paths.

## Backups and restoration

In the app drawer’s **Settings**, create an encrypted backup, download it, and keep a copy away from the machine. Keep the password separately. The archive contains canonical lived state, workspace files/artifacts, appearance assets, and exact application source/lockfiles for reconstruction. Dedicated credential stores, session secrets, recognized model weights, derived indexes, and predecessor archives are excluded. Arbitrary workspace files can still contain embedded credentials.

Exports checkpoint supported mutations. They are application backups, not bootable OS images or copies of unrelated host services/files. Predecessor archives have a separate export control. Server-side download copies expire after up to twenty-four hours; restore previews expire after thirty minutes.

Restoration decrypts and validates an archive before presenting a preview. Commit requires confirmation, a new destination, explicit authority selection, and acknowledgment that the source installation is stopped. Resident identity persists; installation/session identity rotates. Credentials must be reconnected. Index rebuilding requires a configured optional index. Old work is not automatically resumed.

For offline recovery from a working checkout/environment:

```bash
.venv/bin/python -m backend.runtime_cli restore \
  --archive /path/home.tamagi.tar.gz.gpg --destination /new/instance \
  --authority guarded --source-stopped
```

The CLI prompts for the archive password. Inspect the returned configuration and continuity-gap report, then start it with `serve`. Preview does not execute uploaded source; reconstructing a custom release can require its retained source, dependencies, and external services. A configured supervisor can follow a UI-committed restoration; otherwise the UI identifies the configuration to start manually.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Endpoint errors | `llm.base_url`, exact model name, credentials, server reachability, and tool-call support. A local readiness response does not prove the model is available. |
| No autonomous entry | Onboarding, active hours/timezone, chosen rest/wake event, conversation pause, budgets, and Journal’s activity reason. |
| A browser disconnect hid work | Reopen the original conversation and recover the retained request/result. Backend restart is an interruption boundary. |
| Created file is absent from Inventory | Confirm the active instance workspace and place it under `workshop/`; see [workspace conventions](workspace-layout.md). |
| New tool is not callable | Files alone do not activate it. Inspect its proposal, worker trial, evaluation job, and activation. |
| Changed source is not live | A supervised service uses the selected release. Inspect its release identity before assuming the checkout is running. |
| Graph write “succeeded” but is missing | The known graph persistence bug remains open. Inspect actual saved state and retain backups; do not infer durability from that response. |

The unauthenticated local `GET /api/health/ready` endpoint reports application/data readiness, a source-path release label, and model-availability observations. For a supervised release, that label identifies the selected built release; it is not a Git revision for an ordinary checkout. Recovery and operational evidence are described in [implementation and evidence](platform-roadmap.md).
