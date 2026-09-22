"""
config.py — Configuration settings for the lightweight Telegram robot-arm agent.
"""

import os
from pathlib import Path

# Base directory for the Mini Openclaw module
BASE_DIR = Path(__file__).resolve().parent

# Automatically load a local .env file if present
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                val = v.strip().strip('"').strip("'")
                if val:
                    os.environ.setdefault(k.strip(), val)

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Ollama LLM Service Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b").strip()
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

# Robot Arm Flask REST Server URL
# Default local 127.0.0.1:8765, or e.g. http://172.21.96.1:8765
FLASK_URL = os.getenv("FLASK_URL", "http://127.0.0.1:8765").rstrip("/")

# Flask timeout — physical actions (pick, place) can take 30-40 s of real time
# in the current simulation.  60 s avoids premature timeout failures.
FLASK_TIMEOUT_SECONDS = int(os.getenv("FLASK_TIMEOUT_SECONDS", "60"))

# Reachable Envelope (physical safety limits in meters)
# Coordinates outside this envelope are strictly REJECTED (never silently clamped).
REACHABLE_ENVELOPE = {
    "x": (-0.30, 0.30),      # lateral range (left/right)
    "y": (-0.35, -0.05),     # forward range (SO-100 reaches forward along negative Y)
    "z": (0.00, 0.35),       # vertical height above tabletop
}

# Gripper range (0.0 = fully open, 1.0 = fully closed)
GRIPPER_RANGE = (0.0, 1.0)

# Valid named scenarios
VALID_SCENARIOS = {"A", "B", "C"}

# Known workspace landmark targets (semantic world model)
# The agent can refer to these by name; resolve_semantic_target() expands them.
KNOWN_POSITIONS = {
    "right_pad":   {"x": 0.15,  "y": -0.18, "z": 0.015},
    "left_pad":    {"x": -0.15, "y": -0.18, "z": 0.015},
    "home_hover":  {"x": 0.0,   "y": -0.20, "z": 0.16},
}

# ---------------------------------------------------------------------------
# Agent loop limits (Phase 8)
# ---------------------------------------------------------------------------
# MAX_AGENT_ROUNDS: maximum Ollama reasoning invocations per user request.
# The model can observe tool results and plan further within this budget.
MAX_AGENT_ROUNDS = int(os.getenv("MAX_AGENT_ROUNDS", "6"))

# MAX_TOOL_CALLS_PER_TURN: maximum physical tool calls per user request.
# Prevents runaway tool-call chains regardless of how many reasoning rounds
# the model uses.  High-level skills (pick, place) count as 1 each.
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "4"))

# Kept for backward compatibility with any code that may reference the old name.
# New code should use MAX_AGENT_ROUNDS and MAX_TOOL_CALLS_PER_TURN.
MAX_TOOL_STEPS = MAX_AGENT_ROUNDS

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
TOOLS_SCHEMA_PATH = BASE_DIR / "tools_schema.json"
SOUL_FILE_PATH = BASE_DIR / "SOUL.md"
LOG_FILE_PATH = BASE_DIR / "turns.jsonl"
