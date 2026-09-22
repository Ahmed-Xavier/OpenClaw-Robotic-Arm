"""
launcher.py — XavierClaw interactive launcher and control console.

Responsibilities:
  - Orchestrate startup of Ollama, Flask/MuJoCo, and Telegram bot
  - Own the lifecycle of all services (start, stop, restart)
  - Provide an interactive robot> command console (sole caller of input())
  - Broadcast runtime events through EventBus for /logs and /convo monitors

Architecture:
  launcher.py
      ├── EventBus          — thread-safe event broadcaster
      ├── ServiceManager    — process/thread lifecycle
      └── TerminalConsole   — interactive prompt (sole caller of input())
            ├── ConvoMonitor (borrows main-thread input temporarily)
            └── LogMonitor  (borrows main-thread input temporarily)

Concurrency:
  Main thread       → TerminalConsole (the only input() caller)
  Thread: Telegram  → TelegramBotRunner.start() long-poll loop
  Thread: Chat-*    → per-chat worker queues (from bot.py)
  Subprocess        → server.py  (Flask + MuJoCo window)

Windows note: msvcrt is used for non-blocking key detection in monitors,
allowing events to print immediately without waiting for Enter.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Windows non-blocking key input
# ---------------------------------------------------------------------------

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

# ---------------------------------------------------------------------------
# Path bootstrap — add Mini Openclaw to sys.path
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_MINI = _ROOT / "Mini Openclaw"
if str(_MINI) not in sys.path:
    sys.path.insert(0, str(_MINI))

from agent import RobotArmAgent        # noqa: E402
from bot import (                      # noqa: E402
    TelegramBotRunner,
    check_telegram_connectivity,
)
from config import (                   # noqa: E402
    FLASK_URL,
    MAX_AGENT_ROUNDS,
    MAX_TOOL_CALLS_PER_TURN,
    OLLAMA_MODEL,
    OLLAMA_URL,
    TELEGRAM_BOT_TOKEN,
)

_LOG_DIR = _ROOT / "logs"
_SERVER_LOG = _LOG_DIR / "server.log"
_TURNS_LOG = _MINI / "turns.jsonl"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("launcher")

# ---------------------------------------------------------------------------
# Event system
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Event:
    """A runtime event emitted by agents, bots, or the launcher itself.

    type values:
        system       — startup/shutdown milestones
        telegram_in  — user message received from Telegram
        telegram_out — reply sent to Telegram
        terminal_in  — user typed at robot> prompt
        terminal_out — agent replied to terminal user
        tool_call    — agent about to execute a tool
        tool_result  — tool execution finished
        agent_start  — process_message() began (busy-tracking)
        agent_end    — process_message() finished (busy-tracking)
        agent_message— agent final reply ready
        error        — any error
    """
    type: str
    timestamp: str
    source: str        # "telegram", "terminal", "agent", "system"
    data: Dict[str, Any]

    def short_time(self) -> str:
        """Return HH:MM:SS extracted from the ISO timestamp."""
        try:
            return self.timestamp[11:19]
        except Exception:
            return "??:??:??"


class EventBus:
    """Thread-safe in-process event broadcaster.

    Design (plan §6):
      - Snapshot the subscriber list under a lock, then release the lock
        before invoking any callback.
      - A crashing subscriber is logged and skipped; it never propagates.
      - Callers may safely subscribe/unsubscribe from inside a callback.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[Event], None]] = []
        self._log = logging.getLogger("EventBus")

    def subscribe(self, callback: Callable[[Event], None]) -> Callable:
        with self._lock:
            self._subscribers.append(callback)
        return callback

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def emit(self, type_: str, source: str, data: Dict[str, Any]) -> None:
        event = Event(
            type=type_,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            data=data,
        )
        # Snapshot under lock, invoke without lock (plan §6)
        with self._lock:
            snapshot = list(self._subscribers)
        for cb in snapshot:
            try:
                cb(event)
            except Exception as exc:
                self._log.warning("Subscriber %r raised: %s", cb, exc)


# ---------------------------------------------------------------------------
# ServiceManager
# ---------------------------------------------------------------------------

class ServiceManager:
    """Owns the lifecycle of all services.

    Ownership hierarchy (plan §4):
        Flask subprocess — owns MuJoCo; NEVER restarted by launcher
        RobotArmAgent   — LLM + tool routing; recreated on /restart or /model
        TelegramBotRunner — exactly one polling thread at a time (plan §3)
    """

    FLASK_START_TIMEOUT = 40   # seconds to wait for Flask /state to respond
    OLLAMA_START_TIMEOUT = 20  # seconds to wait for ollama serve to come up

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._flask_proc: Optional[subprocess.Popen] = None
        self._agent: Optional[RobotArmAgent] = None
        self._bot_runner: Optional[TelegramBotRunner] = None
        self._bot_thread: Optional[threading.Thread] = None
        self._current_model: str = OLLAMA_MODEL
        self._telegram_diag: Optional[Dict[str, Any]] = None

        # Busy-tracking: count of active process_message() calls in flight
        self._active_count = 0
        self._active_lock = threading.Lock()

        # Subscribe to agent lifecycle events for busy tracking
        bus.subscribe(self._on_lifecycle_event)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> str:
        return self._current_model

    @property
    def agent(self) -> Optional[RobotArmAgent]:
        return self._agent

    @property
    def telegram_connected(self) -> bool:
        return (
            self._bot_runner is not None
            and getattr(self._bot_runner, "running", False)
        )

    def is_robot_busy(self) -> bool:
        """True while any process_message() call is in flight (plan §8)."""
        with self._active_lock:
            return self._active_count > 0

    # ------------------------------------------------------------------
    # Busy tracking via EventBus (plan §8)
    # ------------------------------------------------------------------

    def _on_lifecycle_event(self, event: Event) -> None:
        if event.type == "agent_start":
            with self._active_lock:
                self._active_count += 1
        elif event.type == "agent_end":
            with self._active_lock:
                self._active_count = max(0, self._active_count - 1)

    # ------------------------------------------------------------------
    # Ollama health checks (plan §1 — three stages)
    # ------------------------------------------------------------------

    def ollama_reachable(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
            return r.status_code == 200
        except Exception:
            return False

    def start_ollama_server(self) -> Tuple[bool, str]:
        """Try to launch `ollama serve` and wait for it to come up."""
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False, "ollama not found on PATH"
        except Exception as exc:
            return False, str(exc)

        deadline = time.time() + self.OLLAMA_START_TIMEOUT
        while time.time() < deadline:
            time.sleep(1)
            if self.ollama_reachable():
                return True, "OK"
        return False, f"ollama did not start within {self.OLLAMA_START_TIMEOUT}s"

    def ollama_model_exists(self) -> Tuple[bool, str]:
        """Stage 2: verify the configured model is listed in Ollama."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            return False, f"could not list models: {exc}"

        model = self._current_model
        # Match with or without tag suffix
        found = any(
            m == model or m.split(":")[0] == model.split(":")[0]
            for m in models
        )
        if not found:
            available = ", ".join(models) if models else "(none)"
            return False, f"not found. Available: {available}"
        return True, "OK"

    def ping_model(self, model: Optional[str] = None) -> Tuple[bool, str]:
        """Stage 3 / plan §9 step 2: lightweight inference ping."""
        target = model or self._current_model
        payload = {
            "model": target,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1},
        }
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=90,
            )
            if r.status_code == 200 and "message" in r.json():
                return True, "OK"
            return False, f"HTTP {r.status_code}"
        except requests.exceptions.Timeout:
            return False, "timed out"
        except Exception as exc:
            return False, str(exc)

    def list_ollama_models(self) -> List[str]:
        """Return available model names or empty list on error."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Flask / MuJoCo subprocess
    # ------------------------------------------------------------------

    def start_flask(self) -> Tuple[bool, str]:
        """Launch server.py as a subprocess; wait until /state responds."""
        _LOG_DIR.mkdir(exist_ok=True)
        server_py = _ROOT / "server.py"
        if not server_py.exists():
            return False, f"server.py not found at {server_py}"

        try:
            log_fh = open(_SERVER_LOG, "a", encoding="utf-8")
            self._flask_proc = subprocess.Popen(
                [sys.executable, str(server_py)],
                stdout=log_fh,
                stderr=log_fh,
                cwd=str(_ROOT),
            )
        except Exception as exc:
            return False, str(exc)

        deadline = time.time() + self.FLASK_START_TIMEOUT
        while time.time() < deadline:
            if self._flask_proc.poll() is not None:
                return False, "server.py exited unexpectedly (check logs/server.log)"
            try:
                r = requests.get(f"{FLASK_URL}/state", timeout=3)
                if r.status_code == 200 and "eef" in r.json():
                    self._bus.emit("system", "launcher", {"msg": "Flask ready"})
                    return True, "OK"
            except Exception:
                pass
            time.sleep(1)

        return False, f"Flask did not respond within {self.FLASK_START_TIMEOUT}s"

    def flask_running(self) -> bool:
        if self._flask_proc is None:
            return False
        return self._flask_proc.poll() is None

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def _create_agent(self) -> RobotArmAgent:
        return RobotArmAgent(ollama_model=self._current_model)

    # ------------------------------------------------------------------
    # Telegram bot
    # ------------------------------------------------------------------

    def check_telegram(self) -> Dict[str, Any]:
        """Perform a layered diagnostic check of the Telegram connection."""
        diag = check_telegram_connectivity(TELEGRAM_BOT_TOKEN)
        self._telegram_diag = diag
        return diag

    def start_bot(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate connectivity and start the Telegram bot polling thread."""
        if self._agent is None:
            self._agent = self._create_agent()

        diag = self.check_telegram()
        if diag.get("category") == "NO_TOKEN":
            return True, "no Telegram token — Telegram disabled", diag

        if diag.get("category") != "OK":
            return False, diag.get("reason", "connectivity check failed"), diag

        bot_info = diag.get("bot_info") or {}
        username = bot_info.get("username", "RobotArmBot")

        self._bot_runner = TelegramBotRunner(
            TELEGRAM_BOT_TOKEN,
            self._agent,
            event_bus=self._bus,
        )
        self._bot_thread = threading.Thread(
            target=self._bot_runner.start,
            daemon=True,
            name="TelegramPolling",
        )
        self._bot_thread.start()

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self._bot_thread.is_alive():
                break
            if self._bot_runner.running:
                break
            time.sleep(0.1)

        if self._bot_thread.is_alive() and self._bot_runner.running:
            self._bus.emit("system", "launcher", {"msg": f"Telegram connected as @{username}"})
            return True, f"OK (@{username})", diag
        else:
            return False, "bot runner failed to stay in polling state", diag

    # ------------------------------------------------------------------
    # Restart (plan §3 — strict single-runner guarantee)
    # ------------------------------------------------------------------

    def restart_agent(self, new_model: Optional[str] = None) -> None:
        """Restart the agent software only. Flask/robot state untouched (plan §7).

        Sequence:
          1. stop old Telegram runner
          2. join old polling thread — confirm it stopped
          3. create new RobotArmAgent (same Flask, same robot state)
          4. create new TelegramBotRunner
          5. start exactly one new polling thread
        """
        # 1. Stop old runner
        if self._bot_runner is not None:
            self._bot_runner.stop()

        # 2. Join — confirm the old thread is dead (plan §3)
        if self._bot_thread is not None:
            self._bot_thread.join(timeout=20)
            if self._bot_thread.is_alive():
                print("  [warning] old polling thread did not stop within 20s;"
                      " it will die naturally as a daemon.")

        # 3. Apply new model if requested
        if new_model:
            self._current_model = new_model

        # 4. Create fresh agent (does NOT touch Flask or RobotAPI state)
        self._agent = self._create_agent()

        # 5. Start exactly one new polling thread
        if TELEGRAM_BOT_TOKEN:
            diag = self.check_telegram()
            if diag.get("category") == "OK":
                self._bot_runner = TelegramBotRunner(
                    TELEGRAM_BOT_TOKEN,
                    self._agent,
                    event_bus=self._bus,
                )
                self._bot_thread = threading.Thread(
                    target=self._bot_runner.start,
                    daemon=True,
                    name="TelegramPolling",
                )
                self._bot_thread.start()
            else:
                self._bot_runner = None
                self._bot_thread = None
                print(f"  [Telegram offline: {diag.get('reason')}]")
        else:
            self._bot_runner = None
            self._bot_thread = None

        self._bus.emit("system", "launcher", {
            "msg": f"agent restarted with model={self._current_model}",
            "model": self._current_model,
        })

    # ------------------------------------------------------------------
    # Model change (plan §9 — atomic validation)
    # ------------------------------------------------------------------

    def set_model(self, model: str) -> None:
        """Apply a pre-validated new model and restart agent cleanly."""
        self.restart_agent(new_model=model)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop_all(self) -> None:
        """Gracefully stop Telegram runner, then Flask subprocess."""
        if self._bot_runner is not None:
            self._bot_runner.stop()
        if self._bot_thread is not None:
            self._bot_thread.join(timeout=20)

        if self._flask_proc is not None and self._flask_proc.poll() is None:
            self._flask_proc.terminate()
            try:
                self._flask_proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self._flask_proc.kill()

        self._bus.emit("system", "launcher", {"msg": "shutdown complete"})


# ---------------------------------------------------------------------------
# Monitor helpers
# ---------------------------------------------------------------------------

def _format_convo_event(event: Event) -> Optional[str]:
    """Format an event for /convo display. None = skip."""
    t = event.short_time()
    d = event.data
    if event.type == "telegram_in":
        return f"[{t}] [Telegram → Agent]\n  {d.get('text', '')}"
    if event.type == "telegram_out":
        return f"[{t}] [Agent → Telegram]\n  {d.get('reply', '')}"
    if event.type == "terminal_in":
        return f"[{t}] [Terminal → Agent]\n  {d.get('text', '')}"
    if event.type == "terminal_out":
        return f"[{t}] [Agent → Terminal]\n  {d.get('reply', '')}"
    if event.type == "tool_call":
        args = json.dumps(d.get("arguments", {}), ensure_ascii=False)
        return f"[{t}] [Tool]  {d.get('name', '?')}({args})"
    if event.type == "tool_result":
        ok = d.get("success", "?")
        detail = f"  {d.get('detail', '')}" if d.get("detail") else ""
        return f"[{t}] [Tool result]  success={ok}{detail}"
    if event.type == "agent_message":
        return f"[{t}] [Agent]\n  {d.get('content', '')}"
    if event.type in ("system", "error"):
        return f"[{t}] [{event.type.upper()}] {d.get('msg', '')}"
    return None


def _format_log_event(event: Event) -> Optional[str]:
    """Format an event for /logs display. None = skip."""
    t = event.short_time()
    d = event.data
    if event.type == "telegram_in":
        return f"[{t}] Telegram in:   {d.get('text', '')[:70]}"
    if event.type == "telegram_out":
        return f"[{t}] Telegram out:  {d.get('reply', '')[:70]}"
    if event.type == "terminal_in":
        return f"[{t}] Terminal in:   {d.get('text', '')[:70]}"
    if event.type == "tool_call":
        args = json.dumps(d.get("arguments", {}), ensure_ascii=False)
        return f"[{t}] Tool call:     {d.get('name', '?')}({args})"
    if event.type == "tool_result":
        ok = "success" if d.get("success") else "FAILED"
        return f"[{t}] Tool result:   {ok}"
    if event.type == "agent_message":
        return f"[{t}] Agent reply ready"
    if event.type in ("system", "error"):
        return f"[{t}] {event.type.upper():8} {d.get('msg', '')}"
    return None


def _monitor_wait_for_quit() -> None:
    """Block until the user types 'q' (+ Enter on non-Windows).

    On Windows uses msvcrt for non-blocking key reads so events can
    print immediately without waiting for Enter.
    On other platforms falls back to input() — user must press Enter.
    """
    if _HAS_MSVCRT:
        buf = ""
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    if buf.strip().lower() == "q":
                        return
                    buf = ""
                elif ch == "\x03":  # Ctrl+C
                    return
                else:
                    buf += ch
            time.sleep(0.05)
    else:
        while True:
            try:
                line = input("").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return
            if line == "q":
                return


class ConvoMonitor:
    """Live conversation monitor.

    Borrows the main thread temporarily. Events print immediately as they
    arrive from Telegram/agent threads (via EventBus callbacks).
    """

    def __init__(self, bus: EventBus):
        self._bus = bus

    def _on_event(self, event: Event) -> None:
        line = _format_convo_event(event)
        if line:
            print(line, flush=True)

    def run(self) -> None:
        print()
        print("=" * 54)
        print("  LIVE CONVERSATION")
        print("  Type q + Enter to return")
        print("=" * 54)
        print()
        self._bus.subscribe(self._on_event)
        try:
            _monitor_wait_for_quit()
        finally:
            self._bus.unsubscribe(self._on_event)
            print("\n  Returning to console...\n")


class LogMonitor:
    """Live log viewer.

    Shows the last N turns from turns.jsonl as history, then switches to
    live EventBus events. Does NOT poll the file — the bus is the live source.
    """

    HISTORY_LINES = 10

    def __init__(self, bus: EventBus):
        self._bus = bus

    def _on_event(self, event: Event) -> None:
        line = _format_log_event(event)
        if line:
            print(f"  {line}", flush=True)

    def _show_history(self) -> None:
        if not _TURNS_LOG.exists():
            return
        try:
            raw_lines = _TURNS_LOG.read_text(encoding="utf-8").splitlines()
            recent = raw_lines[-self.HISTORY_LINES:]
            if not recent:
                return
            print("  --- Recent history ---")
            for raw in recent:
                try:
                    rec = json.loads(raw)
                    ts = rec.get("timestamp", "")[:19].replace("T", " ")
                    user = rec.get("user_message", "?")[:60]
                    reply = rec.get("final_reply", "?")[:60]
                    print(f"  [{ts}] User:  {user}")
                    print(f"               Agent: {reply}")
                except Exception:
                    pass
            print()
        except Exception:
            pass

    def run(self) -> None:
        print()
        print("=" * 54)
        print("  LIVE LOGS")
        print("  Type q + Enter to return")
        print("=" * 54)
        print()
        self._show_history()
        print("  --- Live events ---\n")
        self._bus.subscribe(self._on_event)
        try:
            _monitor_wait_for_quit()
        finally:
            self._bus.unsubscribe(self._on_event)
            print("\n  Returning to console...\n")


# ---------------------------------------------------------------------------
# TerminalConsole — the interactive robot> prompt
# ---------------------------------------------------------------------------

# Command registry: cmd → (description, method_name)
# Generated from here so /help is always in sync.
_COMMANDS: Dict[str, Tuple[str, str]] = {
    "/help":    ("Show this help",                         "_cmd_help"),
    "/status":   ("Show agent / robot / service status",    "_cmd_status"),
    "/telegram": ("Show Telegram connectivity & status",     "_cmd_telegram"),
    "/logs":     ("Live log viewer  (q to exit)",           "_cmd_logs"),
    "/convo":    ("Live conversation monitor  (q to exit)", "_cmd_convo"),
    "/model":    ("Show or change the active model",        "_cmd_model"),
    "/restart":  ("Restart the agent",                      "_cmd_restart"),
    "/stop":     ("Gracefully stop everything",             "_cmd_stop"),
    "/clear":    ("Clear the terminal",                     "_cmd_clear"),
    "/exit":     ("Shutdown and exit",                      "_cmd_exit"),
}

# Fixed chat_id for terminal sessions — distinct from any Telegram chat_id
TERMINAL_CHAT_ID = 0


class TerminalConsole:
    """Interactive robot> prompt loop.

    This is the ONLY place in the process that calls input().
    Monitors borrow the main thread's input() while active (plan §2).
    """

    def __init__(self, service: ServiceManager, bus: EventBus):
        self._service = service
        self._bus = bus
        self._running = True

    # ------------------------------------------------------------------
    # Main loop — sole input() caller in this process
    # ------------------------------------------------------------------

    def run(self) -> None:
        while self._running:
            try:
                line = input("robot> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                self._shutdown(ask=True)
                break
            if not line:
                continue
            if line.startswith("/"):
                self._dispatch(line)
            else:
                self._send_to_agent(line)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> None:
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None

        entry = _COMMANDS.get(cmd)
        if entry is None:
            print(f"  Unknown command: {cmd}")
            print("  Type /help for available commands.")
            return

        method = getattr(self, entry[1])
        if cmd in ("/model", "/telegram"):
            method(arg)
        else:
            method()

    # ------------------------------------------------------------------
    # Normal message → agent
    # ------------------------------------------------------------------

    def _send_to_agent(self, text: str) -> None:
        agent = self._service.agent
        if agent is None:
            print("  [error] Agent not running. Try /restart.")
            return

        self._bus.emit("terminal_in", "terminal", {"text": text})
        self._bus.emit("agent_start", "terminal", {"text": text})

        def _status_cb(event: str, tool_name: str, data: Any) -> None:
            labels = {
                "pick": "Picking up object",
                "place": "Placing object",
                "move_to": "Moving to position",
                "gripper": "Adjusting gripper",
                "reset_home": "Returning to home",
                "state": "Checking state",
                "camera": "Capturing image",
                "scenario": "Running scenario",
                "collisions": "Checking collisions",
            }
            label = labels.get(tool_name, tool_name)
            if event == "started":
                print(f"  [Tool] {label}...")
                self._bus.emit("tool_call", "agent", {
                    "name": tool_name,
                    "arguments": data if isinstance(data, dict) else {},
                })
            elif event == "completed":
                print(f"  [Done] {label}")
                self._bus.emit("tool_result", "agent", {
                    "name": tool_name, "success": True, "detail": "",
                })
            elif event == "failed":
                err = ""
                if isinstance(data, dict):
                    err_d = data.get("error") or {}
                    if isinstance(err_d, dict):
                        err = err_d.get("message", "")
                print(f"  [Fail] {label}")
                self._bus.emit("tool_result", "agent", {
                    "name": tool_name, "success": False, "detail": err,
                })

        print(f"\n  You: {text}\n")
        try:
            result = agent.process_message(
                user_text=text,
                chat_id=TERMINAL_CHAT_ID,
                status_callback=_status_cb,
            )
        except Exception as exc:
            print(f"  [error] Agent raised: {exc}")
            self._bus.emit("error", "terminal", {"msg": str(exc)})
            self._bus.emit("agent_end", "terminal", {})
            return

        reply = result.get("reply", "")
        rounds = result.get("agent_rounds", "?")
        tool_count = result.get("physical_tool_calls", "?")
        failure = result.get("failure_type")

        print(f"\n  Army: {reply}")
        print(f"  (rounds: {rounds}  |  tool calls: {tool_count})\n")
        if failure:
            print(f"  [!] {failure}\n")

        self._bus.emit("terminal_out", "agent", {
            "reply": reply, "rounds": rounds, "tool_calls": tool_count,
        })
        self._bus.emit("agent_end", "terminal", {})

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _cmd_help(self) -> None:
        print()
        print("  Available commands:")
        print()
        width = max(len(k) for k in _COMMANDS)
        for cmd, (desc, _) in _COMMANDS.items():
            print(f"    {cmd:<{width + 2}} {desc}")
        print()
        print("  Anything else is sent to Army as a message.")
        print()

    def _cmd_status(self) -> None:
        sm = self._service
        print()
        print("  Agent:")
        print(f"    Name:   Army")
        print(f"    State:  {'BUSY' if sm.is_robot_busy() else 'READY'}")
        print(f"    Model:  ollama/{sm.current_model}")
        print()

        print("  Robot:")
        try:
            r = requests.get(f"{FLASK_URL}/state", timeout=3)
            st = r.json()
            eef = st.get("eef", [0, 0, 0])
            holding = st.get("is_holding", False)
            held = st.get("held_object", "none")
            print(f"    Backend:  MuJoCo")
            print(f"    EEF:      [{eef[0]:.3f}, {eef[1]:.3f}, {eef[2]:.3f}]")
            print(f"    Holding:  {holding}")
            print(f"    Object:   {held if holding else 'none'}")
        except Exception as exc:
            print(f"    [Flask unreachable: {exc}]")
        print()

        print("  Services:")
        flask_ok = sm.flask_running()
        print(f"    Flask API:  {'RUNNING' if flask_ok else 'DOWN'}  ({FLASK_URL})")
        tg_ok = sm.telegram_connected
        tg_diag = getattr(sm, "_telegram_diag", None) or {}
        bot_info = tg_diag.get("bot_info") or {}
        username = bot_info.get("username")
        if tg_ok:
            tg_str = f"CONNECTED (@{username})" if username else "CONNECTED"
        elif tg_diag.get("category") == "NO_TOKEN":
            tg_str = "DISABLED (no token)"
        elif tg_diag.get("category"):
            tg_str = f"OFFLINE ({tg_diag.get('reason', 'unreachable')})"
        else:
            tg_str = "DISCONNECTED"
        print(f"    Telegram:   {tg_str}")
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
            ollama_ok = True
        except Exception:
            ollama_ok = False
        print(f"    Ollama:     {'CONNECTED' if ollama_ok else 'UNREACHABLE'}  ({OLLAMA_URL})")
        print()

        print("  Limits:")
        print(f"    Max agent rounds:  {MAX_AGENT_ROUNDS}")
        print(f"    Max tool calls:    {MAX_TOOL_CALLS_PER_TURN}")
        print()

    def _cmd_telegram(self, arg: Optional[str] = None) -> None:
        sm = self._service
        action = (arg or "").strip().lower()

        if action in ("connect", "retry", "start"):
            if sm.telegram_connected:
                print("\n  Telegram is already CONNECTED and polling.\n")
                return
            print("\n  Testing connectivity and connecting to Telegram...")
            ok, msg, diag = sm.start_bot()
            if ok:
                username = diag.get("bot_info", {}).get("username", "bot")
                print(f"  Connected as @{username}!\n")
            else:
                print(f"  Connection failed: {msg}\n")
            return

        diag = sm.check_telegram()
        polling = "RUNNING" if sm.telegram_connected else "OFFLINE"
        bot_info = diag.get("bot_info") or {}
        username = bot_info.get("username")

        print()
        print("  Telegram")
        print()
        print(f"    Network:        {diag.get('network', 'UNKNOWN')}")
        print(f"    DNS:            {diag.get('dns', 'UNKNOWN')}")
        print(f"    HTTPS:          {diag.get('https', 'UNKNOWN')}")
        print(f"    Bot API:        {diag.get('bot_api', 'UNKNOWN')}")
        print(f"    Authentication: {diag.get('auth', 'UNKNOWN')}")
        poll_str = f"{polling} (@{username})" if (polling == "RUNNING" and username) else polling
        print(f"    Polling:        {poll_str}")
        print()

        if diag.get("category") != "OK":
            print("  Reason:")
            print(f"    {diag.get('reason')}")
            if diag.get("category") in ("DNS_FAILURE", "TIMEOUT", "TLS_FAILURE", "NETWORK_FAILURE", "CONNECTION_FAILURE"):
                print("    The bot token was NOT tested because the Telegram API")
                print("    could not be reached.")
            elif diag.get("category") == "AUTH_FAILURE":
                print("    Telegram rejected the bot token.")
            print()
            if not sm.telegram_connected:
                print("  Tip: Use '/telegram connect' to retry once network is available.")
                print()

    def _cmd_logs(self) -> None:
        LogMonitor(self._bus).run()

    def _cmd_convo(self) -> None:
        ConvoMonitor(self._bus).run()

    def _cmd_model(self, arg: Optional[str]) -> None:
        sm = self._service
        current = sm.current_model

        if arg is None:
            # Display mode: list and optionally prompt for change
            print()
            print(f"  Current model:  ollama/{current}")
            print()
            models = sm.list_ollama_models()
            if models:
                print("  Available models:")
                for m in models:
                    marker = "*" if (
                        m == current or m.split(":")[0] == current.split(":")[0]
                    ) else " "
                    print(f"    [{marker}] {m}")
            else:
                print("  (Could not list models — Ollama may be unreachable)")
            print()
            try:
                new_name = input("  New model (Enter to keep current): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_name:
                print(f"  Model unchanged: {current}")
                return
            arg = new_name

        # Atomic validation pipeline (plan §9)
        print(f"\n  Validating model '{arg}'...")

        # Step 1: check existence
        models = sm.list_ollama_models()
        if models:  # skip existence check if we can't list (Ollama may be unreachable)
            found = any(
                m == arg or m.split(":")[0] == arg.split(":")[0]
                for m in models
            )
            if not found:
                print(f"  Model '{arg}' not found in Ollama.")
                if models:
                    print(f"  Available: {', '.join(models)}")
                print(f"  Current model unchanged: {current}")
                return

        # Step 2: inference ping
        print("  Pinging model (may take a moment)...")
        ok, reason = sm.ping_model(arg)
        if not ok:
            print(f"  Model change rejected: {reason}")
            print(f"  Current model unchanged: {current}")
            return

        # Step 3: apply atomically — only reached if both checks passed
        print(f"  Restarting agent with model '{arg}'...")
        sm.set_model(arg)
        print(f"  Model changed: {current} → {arg}")
        print()

    def _cmd_restart(self) -> None:
        sm = self._service
        # Busy check (plan §8)
        if sm.is_robot_busy():
            print()
            print("  Robot is currently busy.")
            print("  Waiting for current action to finish...")
            deadline = time.time() + 60
            while sm.is_robot_busy() and time.time() < deadline:
                time.sleep(1)
            if sm.is_robot_busy():
                print("  [warning] Still busy after 60s — restarting anyway.")

        print()
        print("  Stopping agent...")
        sm.restart_agent()
        print("  Agent READY.")
        print()

    def _cmd_stop(self) -> None:
        self._shutdown(ask=False)

    def _cmd_clear(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")

    def _cmd_exit(self) -> None:
        self._shutdown(ask=True)

    def _shutdown(self, ask: bool) -> None:
        sm = self._service
        if ask and sm.is_robot_busy():
            print()
            print("  A robot action is currently running.")
            try:
                confirm = input("  Exit anyway? [y/N]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                confirm = "y"
            if confirm != "y":
                print("  Cancelled.")
                return

        print()
        print("  Shutting down...")
        self._running = False
        sm.stop_all()
        print("  Done.")


# ---------------------------------------------------------------------------
# Startup display
# ---------------------------------------------------------------------------

def _step(label: str, width: int = 36) -> None:
    sys.stdout.write(f"  {label:<{width}}")
    sys.stdout.flush()


def _ok(detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"OK{suffix}")


def _fail(reason: str) -> None:
    print(f"FAIL\n  → {reason}")


def _print_banner() -> None:
    print()
    print("=" * 52)
    print("        XAVIERCLAW — ROBOT ARM AGENT")
    print("=" * 52)
    print()


def run_startup(service: ServiceManager) -> bool:
    """Run the ordered startup sequence with aligned status display.

    Returns True if all critical services came up (agent + Flask).
    Telegram failure is non-fatal — terminal still works.
    """
    _print_banner()

    # ---- Ollama: reachability ----
    _step("Checking Ollama...")
    if service.ollama_reachable():
        _ok()
    else:
        print("not running")
        _step("  Starting Ollama server...")
        ok, msg = service.start_ollama_server()
        if ok:
            _ok()
        else:
            _fail(msg)
            return False

    # ---- Ollama: model existence ----
    _step(f"Model {service.current_model}...")
    ok, msg = service.ollama_model_exists()
    if ok:
        _ok()
    else:
        _fail(msg)
        return False

    # ---- Ollama: inference validation ----
    _step("Validating model response...")
    t0 = time.time()
    ok, reason = service.ping_model()
    elapsed = time.time() - t0
    if ok:
        _ok(f"{elapsed:.1f}s")
    else:
        _fail(reason)
        return False

    # ---- Flask / MuJoCo ----
    _step("Starting Flask / MuJoCo...")
    t0 = time.time()
    ok, msg = service.start_flask()
    elapsed = time.time() - t0
    if ok:
        _ok(f"{elapsed:.1f}s")
    else:
        _fail(msg)
        print("  Check logs/server.log for details.")
        return False

    # ---- Telegram bot (non-fatal) ----
    _step("Connecting Telegram...")
    ok, msg, diag = service.start_bot()
    if ok:
        if "disabled" in msg.lower():
            print(f"disabled  (no token)")
            if service.agent is None:
                service._agent = service._create_agent()
        else:
            username = diag.get("bot_info", {}).get("username", "")
            _ok(f"@{username}" if username else "")
    else:
        print("OFFLINE")
        print()
        print(f"  Reason:")
        print(f"    {diag.get('reason', msg)}")
        if diag.get("category") in ("DNS_FAILURE", "TIMEOUT", "TLS_FAILURE", "NETWORK_FAILURE", "CONNECTION_FAILURE"):
            print("    The bot token was NOT tested because the Telegram API")
            print("    could not be reached.")
        elif diag.get("category") == "AUTH_FAILURE":
            print("    Telegram rejected the bot token.")
            print("    Please check TELEGRAM_BOT_TOKEN in Mini Openclaw/.env.")
        print()
        print("  Terminal still available. Telegram is inactive.")
        if service.agent is None:
            service._agent = service._create_agent()

    # ---- Ready ----
    if service.telegram_connected:
        tg_info = getattr(service, "_telegram_diag", None) or {}
        username = tg_info.get("bot_info", {}).get("username")
        tg_status = f"connected (@{username})" if username else "connected"
    elif diag.get("category") == "NO_TOKEN":
        tg_status = "disabled (no token)"
    else:
        tg_status = f"offline ({diag.get('reason', 'unreachable')})"

    print()
    print("=" * 52)
    print(f"  READY  |  Agent: Army  |  Model: {service.current_model}")
    print(f"  Telegram: {tg_status}")
    print("=" * 52)
    print()
    print("  /help for commands.")
    print("  Type a message to talk to Army.")
    print()
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    bus = EventBus()
    service = ServiceManager(bus)

    ready = run_startup(service)
    if not ready:
        print()
        print("  Startup incomplete — some services may not be available.")
        print("  Use /status to inspect state, or /exit to quit.")
        print()
        # Ensure agent exists so terminal is still usable
        if service.agent is None:
            service._agent = service._create_agent()

    console = TerminalConsole(service, bus)
    console.run()


if __name__ == "__main__":
    main()
