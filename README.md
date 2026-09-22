# XavierClaw — SO-100 Robot Arm Simulation

XavierClaw is a local, safety-bounded natural-language controller for an SO-100-style robotic arm simulation. It combines a MuJoCo simulation, a small Flask API, a local Ollama model, and optional Telegram control.

The project is simulation-first. It does not require a cloud AI API, ROS, a database, Docker, or external message brokers.

> **Safety note:** This project can issue physical-style movement commands. Treat Telegram access as control access. Run it only on a trusted machine and network, and review commands before adapting it to real hardware.

## What it includes

- MuJoCo SO-100 simulation with a desktop viewer, camera, pick/place routines, and collision inspection.
- Flask API for robot actions and state.
- Local Ollama agent with tool schemas, a reachable-envelope check, bounded tool calls, and per-chat serial processing.
- Terminal console and optional Telegram bot.
- Bounded Telegram retry/reconnect handling that never reruns an agent turn after a delivery failure.

## Prerequisites

- Windows 10/11 with a desktop session (the default MuJoCo server opens a viewer).
- Python 3.10 or newer. Python 3.11 is the tested version.
- [Ollama](https://ollama.com/) installed and available on `PATH` for natural-language control.
- A Telegram bot token only if Telegram control is wanted.

All Python runtime packages are declared in `requirements.txt`. The MuJoCo model and mesh assets are included in `third_party/`; there are no submodules or additional asset downloads.

## Quick start

From PowerShell:

```powershell
git clone https://github.com/Ahmed-Xavier/OpenClaw-Robotic-Arm.git
cd OpenClaw-Robotic-Arm

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install the local model once.
ollama pull qwen3.5:4b

# Optional: enable Telegram.
Copy-Item "Mini Openclaw\.env.example" "Mini Openclaw\.env"
```

If Telegram is enabled, edit `Mini Openclaw\.env` and set `TELEGRAM_BOT_TOKEN`. Do not commit that file.

Start the complete local experience:

```powershell
python launcher.py
```

`launch.bat` is the equivalent Windows shortcut. The launcher checks Ollama, starts the Flask/MuJoCo service, creates the agent, and starts Telegram when a token is configured. Use `/status` and `/telegram` in the `robot>` console for diagnostics.

### Terminal-only robot API

To run just the simulation API and MuJoCo viewer:

```powershell
python server.py
```

To exercise the simulation’s direct menu without the agent:

```powershell
python robot_api.py
```

## Configuration

`Mini Openclaw/.env` is optional. Blank values use the defaults below.

| Setting | Default | Purpose |
|---|---:|---|
| `TELEGRAM_BOT_TOKEN` | unset | Enables Telegram when provided. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama server. |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Model used for planning and replies. |
| `FLASK_URL` | `http://127.0.0.1:8765` | Robot API endpoint used by the agent. |
| `OLLAMA_TIMEOUT_SECONDS` | `60` | Per-request Ollama timeout. |
| `FLASK_TIMEOUT_SECONDS` | `60` | Physical-action API timeout. |
| `MAX_AGENT_ROUNDS` | `6` | Maximum model reasoning rounds per request. |
| `MAX_TOOL_CALLS_PER_TURN` | `4` | Maximum robot tool calls per request. |
| `MAX_HISTORY_TURNS` | `6` | Conversation history retained per chat. |

The Flask server listens on `0.0.0.0:8765`; `FLASK_URL` should remain reachable from the launcher. There is no authentication on this development API—do not expose port 8765 to an untrusted network.

## Telegram setup

1. Create a bot with Telegram’s BotFather and copy its token.
2. Put the token in `Mini Openclaw/.env`.
3. Run `python launcher.py`.
4. In the console, run `/telegram` to inspect DNS, HTTPS, authentication, and polling status.

Telegram is optional. Without a token, the terminal workflow remains fully usable. The bot uses one polling loop and serializes commands per chat. Transient polling failures reconnect with backoff; outbound delivery retries never recompute the robot action.

## Common commands

At the `robot>` prompt:

```text
/help                 Show console commands
/status               Show robot, Ollama, Flask, and Telegram state
/telegram             Show Telegram diagnostics
/telegram connect     Retry starting Telegram polling
/new                  Reset the terminal conversation
/exit                 Shut down cleanly
```

Examples:

```text
pick up the sphere
place it on the right pad
show me the camera
```

## HTTP API

`server.py` exposes these local endpoints on port 8765:

| Method | Path | Description |
|---|---|---|
| `POST` | `/move_to` | Move to `{"x": ..., "y": ..., "z": ...}`. |
| `POST` | `/pick` | Pick `cube` or `sphere`; optional `{"target": "sphere"}`. |
| `POST` | `/place` | Place at `{"x": ..., "y": ..., "z": ...}`. |
| `POST` | `/gripper` | Set gripper openness. |
| `GET` | `/state` | Read concise robot state. |
| `GET` | `/state/full` | Read MuJoCo debug telemetry. |
| `GET` | `/camera` | Save and return a wrist-camera image path. |
| `POST` | `/reset_home` | Return to the home pose. |
| `GET` | `/collisions` | Report current contact pairs. |
| `POST` | `/scenario` | Run scenario `A`, `B`, or `C`. |

Only run one `server.py` instance at a time. It owns one global MuJoCo viewer and robot state; the Flask reloader is deliberately disabled.

## Tests

Install the test dependency, then run the full mocked suite:

```powershell
python -m pip install pytest
python -m pytest -q "Mini Openclaw"
```

Tests do not require a real Telegram token, Ollama service, or MuJoCo viewer.

## Repository map

```text
launcher.py                 Interactive launcher and service lifecycle
server.py                   Flask API and single MuJoCo RobotAPI instance
robot_api.py                Simulation, IK, pick/place, camera, state
Mini Openclaw/
  agent.py                  Ollama tool-calling agent and safety validation
  bot.py                    Telegram client, queues, retries, reconnects
  config.py                 Environment-backed configuration
  .env.example              Safe configuration template
  tools_schema.json         Tool contract presented to Ollama
third_party/so_arm100/      Vendored model and mesh assets
```

## Troubleshooting

- **`ollama did not start within 20s`:** Start Ollama manually, confirm `ollama list` works, and ensure `qwen3.5:4b` was pulled. Telegram can still poll while Ollama is recovering, but agent requests need Ollama to answer.
- **Telegram shows authenticated but polling offline:** Run `/telegram connect`. If startup previously stopped before Telegram initialization, restart with the current launcher.
- **No MuJoCo viewer or a stale viewer:** Ensure only one `server.py` process is listening on port 8765, then restart the launcher.
- **PowerShell blocks virtual-environment activation:** Run `Set-ExecutionPolicy -Scope Process Bypass` for the current terminal, then activate `.venv` again.

## Limits

This is a local simulation and development API, not a production robot-control system. It has no user authentication, durable job queue, or process-restart deduplication. Do not connect it directly to a physical arm without a separate hardware safety review, emergency-stop design, access control, and physical validation.
