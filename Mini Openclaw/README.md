# Mini Openclaw — Lightweight Telegram-to-Robot-Arm Agent

A minimal-overhead Python agent controlling the MuJoCo SO-100 robot arm via Telegram and local Ollama (`qwen3.5:4b`). Built to replace heavy general-purpose frameworks with zero sandboxing fights and no tool-search latency.

---

## Key Features

- **Direct Tool Schema (`tools_schema.json`)**: Matches all 9 endpoints of `server.py` (`move_to`, `pick`, `place`, `gripper`, `state`, `camera`, `reset_home`, `scenario`, `collisions`).
- **Safety Envelope Validation**: Validates all coordinates against the physical reachable envelope (`X: [-0.30, 0.30]`, `Y: [-0.35, -0.05]`, `Z: [0.00, 0.35]`) and gripper limits *before* contacting the robot. Out-of-bounds requests are rejected with a clear explanation—never silently clamped.
- **Multi-Step Bounded Execution**: Automatically executes multi-step actions (e.g. "pick up the cube and place it on the right") in a bounded loop capped at 4 steps.
- **Serial Queue per Chat**: Incoming messages for the same chat are queued and processed strictly sequentially so conflicting commands never race.
- **Human-Readable Turn Logging**: Every interaction is appended to `turns.jsonl` with timestamps, tool calls, responses, and final replies.
- **Zero Heavy Dependencies**: Uses standard `requests` for both Telegram long-polling and Flask/Ollama communication.

---

## File Structure

```
Mini Openclaw/
├── tools_schema.json    # JSON schema defining the 9 robot arm endpoints
├── config.py            # Envelope boundaries, endpoints, environment config
├── agent.py             # Agent loop: envelope validation, Ollama chat, Flask execution, history, logging
├── bot.py               # Telegram bot runner (with serial per-chat queue & CLI fallback)
├── test_agent.py        # Automated test suite
├── .env.example         # Example environment configuration
└── turns.jsonl          # Runtime interaction log (generated on first run)
```

---

## Quick Start

### 1. Configuration
For new setups, copy `.env.example` to `.env` (or rename it by removing `.example`):
```bash
cp .env.example .env
```
Inside your new `.env` file, fill in your details:
```env
# Required for Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Optional overrides (defaults to local Ollama and local Flask)
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:4b
FLASK_URL=http://127.0.0.1:8765
```
*(Note: `.env.example` is committed to Git as a clean structure template with blank values. Your private `.env` file is ignored by Git and never uploaded).*

### 2. Start the Arm Simulation Server
In the project root, run:
```bash
python server.py
```
*(Or double-click `Run Server.bat`)*

### 3. Run the Agent

**Via Telegram:**
Ensure `TELEGRAM_BOT_TOKEN` is set, then start the bot:
```bash
python bot.py
```

**Interactive CLI Test Mode (no Telegram token needed):**
```bash
python bot.py --cli
```

### 4. Run Automated Tests
```bash
python test_agent.py
```
*(Runs 13 tests covering schema definitions, envelope limits, multi-step caps, logging, and live Ollama tool-calling).*
