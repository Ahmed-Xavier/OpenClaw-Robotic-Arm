# XavierClaw Agent

This folder contains XavierClaw’s local Ollama agent and optional Telegram transport. It is normally started from the repository root with `python launcher.py`.

For full installation, simulation, security, and troubleshooting instructions, see the [root README](../README.md).

## Responsibilities

- Loads `tools_schema.json` and asks local Ollama to choose robot actions.
- Validates coordinates against the configured reachable envelope before contacting the robot API.
- Caps model rounds and robot tool calls for each request.
- Keeps conversation history and a JSONL turn log per local runtime.
- Serializes work per Telegram chat.
- Uses a single Telegram polling loop with bounded reconnect and delivery handling.

## Local configuration

Copy the example file, then add a Telegram token only when Telegram control is required:

```powershell
Copy-Item .env.example .env
```

```env
TELEGRAM_BOT_TOKEN=
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:4b
FLASK_URL=http://127.0.0.1:8765
```

`.env` is ignored by Git. Do not put tokens in source code, issues, logs, or commits.

Additional limits and timeouts are configured through environment variables in `config.py`; the root README lists their defaults.

## Running directly

Run these commands from this directory only after the Flask/MuJoCo API and Ollama are available:

```powershell
# Telegram mode; requires TELEGRAM_BOT_TOKEN.
python bot.py

# Terminal-only agent mode; no Telegram token required.
python bot.py --cli
```

The preferred path is still `python launcher.py` from the repository root because it owns startup and clean shutdown of the services.

## Testing

From the repository root:

```powershell
python -m pip install pytest
python -m pytest -q "Mini Openclaw"
```

The tests use mocks; they do not contact Telegram, Ollama, or a live robot.

## Runtime files

| File | Role |
|---|---|
| `agent.py` | Tool-calling agent, safety checks, history, logging. |
| `bot.py` | Telegram API client, poller, queues, retry/reconnect policy. |
| `config.py` | Environment-backed configuration and safety limits. |
| `SOUL.md` | Agent personality/system instruction. |
| `tools_schema.json` | Tool definitions given to Ollama. |
| `turns.jsonl` | Local generated interaction log; ignored by Git. |

## Telegram delivery behavior

The agent runs once per queued message. If Telegram delivery fails after a robot action has completed, the action is never rerun. Polling and safe/idempotent operations use bounded retries; potentially duplicate visible deliveries are handled conservatively.
