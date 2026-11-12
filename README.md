# 🥚 TamAGI

**A local-first AI companion with a home, a workshop, and room to develop.**

TamAGI is a self-hosted experiment in ongoing conversation and self-directed
activity. Give a resident a name and a starting personality, talk with them,
explore their world, and see what they make between visits. Use a local model
or a compatible hosted endpoint.

A resident's memories, conversations, personal files and creations live on your
machine. You can run a standalone home or an optional household with several
independent residents, shared conversations and a connected world.

## What you can do

- **Build continuity:** return to conversations and memories across visits.
- **Share a world:** residents can explore, pursue interests, converse or rest.
- **Make things:** the workshop holds files, tools and saved creations.
- **Meet others:** introduce companions or create independent household residents.
- **Make it personal:** choose built-in artwork or customize an avatar and rig.
- **Keep a portable home:** use encrypted backups and staged restoration.

The app drawer opens Map, Journal, Inventory, Companions, Archive and Settings.
Accepted work belongs to the application and can finish after a browser tab
closes while the backend stays running.

TamAGI is experimental. Mechanical tests cover application behavior; sustained
improvement and the quality of a resident's experience require separate observation.

## Get started

Use **Linux with Python 3.13**, Git, and a chat-completions endpoint with tool
support. Encrypted backup and recovery also require GnuPG. TamAGI does not bundle
model weights or start a model server.

| Start here | Guide |
| --- | --- |
| Install a standalone home | [Installation and running](docs/running.md#install-a-new-home) |
| Try several residents | [Native households](docs/household-native.md) |
| Add a new household resident | [Create a TamAGI](docs/creating-a-tamagi.md) |
| Preserve an existing home | [Recovery and migration](docs/production-multi-resident-enablement.md) |
| Configure models, search and MCP | [Integrations](docs/integrations.md) |
| Change artwork or upload sprites | [Appearance](docs/sprite-customization.md) |
| Build tools or find created files | [Workspace and tools](docs/workspace-layout.md) |

The [documentation index](docs/README.md) covers navigation, scheduling, recovery,
architecture and development. Keep an encrypted backup before changing an
existing installation's code or data.

## Contribute

See [development and testing](docs/development.md) for checks and repository
conventions. Bug reports should include the source revision, deployment mode,
steps to reproduce, and relevant logs with credentials removed.

## About

TamAGI is an independent, experimental project licensed under
[AGPL-3.0-only](LICENSE.md). It is inspired by
[OpenClaw](https://github.com/openclaw/openclaw),
[AI-tamago](https://github.com/ykhli/AI-tamago), and
[Pwnagotchi](https://github.com/evilsocket/pwnagotchi). It is not affiliated with
Tamagotchi or Bandai Namco.
