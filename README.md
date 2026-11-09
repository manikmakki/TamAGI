# 🥚 TamAGI

**A local-first AI companion with a home, a workshop, and room to develop.**

TamAGI is a self-hosted experiment in ongoing conversation and self-directed activity. Chat with a resident, explore their world, invite companions, and see what they make between visits. Your model can run locally or use a compatible hosted endpoint.

The v0.3 home platform is undergoing burn-in on the `codex/dependable-home` branch. The default `main` branch still contains the older runtime; the setup below selects the home-platform branch. An opt-in [native multi-resident candidate](docs/household-native.md) is available on `codex/adr-0019-household`, with shared conversations, scheduled world turns and individual recovery. Production adoption and PWA push remain [separate work](docs/multi-resident-roadmap.md), and sustained behavioral improvement is still being evaluated.

## What you can try

- **Chat and continuity:** conversations and memories persist in a local SQLite database.
- **A living world:** scheduled opportunities to explore, work on an interest, or choose rest. The World Thread records activity in Journal.
- **A workshop:** Inventory discovers creations in the resident’s `workshop/` folder and distinguishes current files from saved versions.
- **Companions:** durable characters with their own memories, active when invited into a conversation or task.
- **Tools:** local file/command tools, optional MCP integrations, and versioned generated tools.
- **Appearance and Growth:** Vex, Echo, Ember, custom PNGs and rigs, plus deterministic recognition of conversation, continuity, and linked corrections.
- **Portable recovery:** encrypted backups and staged restoration, with application releases separate from lived state.

The app drawer opens **Map, Journal, Inventory, Companions, Archive, and Settings**. Accepted chat work can finish while you browse or close the page, provided the backend stays running. Restarting the backend interrupts work; retained results and receipts remain inspectable.

## Run a new home

The documented environment is **Linux with Python 3.13**. You also need Git, Python’s venv/pip support, and a model endpoint that supports chat completions and tool calls. Install GnuPG (`gpg`) for encrypted backups and the recovery checks. TamAGI does not install or start a model server.

Run as an ordinary user for the `guarded` profile. It applies application policies; it is not an OS sandbox. If you already have a resident, start with [preservation and cutover](docs/running.md#preserve-an-existing-resident).

```bash
git clone --branch codex/dependable-home https://github.com/manikmakki/TamAGI.git
cd TamAGI

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .

python -m backend.runtime_cli fresh --authority guarded \
  --instance "$PWD/data/instances/home" --port 7741
```

Edit `data/instances/home/config.yaml` with your endpoint and an installed model’s exact name. For example, an Ollama endpoint can use:

```yaml
llm:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
  model: "YOUR_INSTALLED_MODEL"
  max_tokens: 2048
  timeout: 180
```

Then start TamAGI:

```bash
python -m backend.runtime_cli serve --config "$PWD/data/instances/home/config.yaml"
```

Open **http://localhost:7741/login**, set your operator password, and complete the introduction. The generated configuration binds to localhost. Stop with Ctrl+C; use the same `serve` command to return to that resident.

SQLite memory works without an embedding model or vector database. Your instance directory contains its state and workspace; keep it when updating application code. Download an encrypted backup from **Settings** and retain a copy elsewhere.

## Choose your next step

| Want to… | Read |
| --- | --- |
| Configure scheduling, networking, services, Docker, or an existing installation | [Running and recovery](docs/running.md) |
| Connect search, MCP tools, optional memory indexes, or credentials | [Integrations and memory](docs/integrations.md) |
| Find workspace files or build a callable tool | [Workspace and tool development](docs/workspace-layout.md) |
| Change artwork, proportions, or uploaded sprites | [Appearance guide](docs/sprite-customization.md) |
| Understand navigation, accepted work, and restart behavior | [Navigation and work](docs/navigation-and-work.md) |
| Check implementation evidence, limitations, or future work | [Documentation index](docs/README.md) |

## Contributing

Bug reports are welcome. Include the branch/commit, platform or legacy mode, deployment method, steps to reproduce, and relevant logs with credentials removed. The [graph persistence issue](https://github.com/manikmakki/TamAGI/issues/5) remains open; a successful legacy graph-write response does not reliably prove persistence.

For development checks:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

The suite uses disposable fixtures; passing it does not complete the seven-day behavioral pilot. See [implementation and evidence](docs/platform-roadmap.md) for the limits of current claims.

## About

TamAGI is an independent, experimental project licensed under [AGPL-3.0-only](LICENSE.md). It is inspired by [OpenClaw](https://github.com/openclaw/openclaw), [AI-tamago](https://github.com/ykhli/AI-tamago), and [Pwnagotchi](https://github.com/evilsocket/pwnagotchi). It is not affiliated with Tamagotchi or Bandai Namco.
