"""
bot.py — Lightweight Telegram Bot Runner for the SO-100 Robot Arm Agent.

Features:
- Direct Telegram Bot API long-polling (zero external framework overhead)
- Per-chat serial queue ensuring sequential execution of commands
- Phase 11: /new handled outside LLM — clears conversation history directly
- Phase 16: Typing keep-alive indicator + single editable status message
             Status events come from actual execution, NOT from Qwen
- Phase 17: /controls command with inline keyboard for manual robot control
             Button actions call Flask directly (bypass Qwen for determinism)
- Sends replies and camera photos
- Graceful error boundary: never crashes on network drops or model issues
- Interactive terminal fallback mode for instant local testing without Telegram
"""

import argparse
import logging
import os
import queue
import socket
import ssl
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from agent import RobotArmAgent
from config import FLASK_TIMEOUT_SECONDS, FLASK_URL, TELEGRAM_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TelegramArmBot")


# ---------------------------------------------------------------------------
# Phase 17 — Manual button step size (meters) for directional buttons
# ---------------------------------------------------------------------------
BUTTON_STEP_XY = 0.03   # lateral / forward step per button press
BUTTON_STEP_Z = 0.03    # vertical step per button press


# ---------------------------------------------------------------------------
# Telegram Diagnostic & Connectivity Classification
# ---------------------------------------------------------------------------

def check_telegram_connectivity(token: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Perform a layered diagnostic check of the Telegram connection path.

    Evaluates:
      - Token presence
      - DNS resolution of api.telegram.org
      - TCP connection to api.telegram.org:443
      - TLS/SSL handshake
      - Telegram Bot API getMe authentication

    Returns a structured dictionary without ever exposing the bot token.
    """
    if not token or not token.strip():
        return {
            "network": "OK",
            "dns": "NOT TESTED",
            "https": "NOT TESTED",
            "bot_api": "NOT TESTED",
            "auth": "NOT TESTED",
            "category": "NO_TOKEN",
            "bot_info": None,
            "reason": "No Telegram bot token configured.",
            "detail": "Set TELEGRAM_BOT_TOKEN in Mini Openclaw/.env to enable Telegram.",
        }

    host = "api.telegram.org"
    port = 443

    # 1. DNS resolution
    try:
        addrinfo = socket.getaddrinfo(host, port)
        if not addrinfo:
            raise socket.gaierror("No address returned")
    except socket.gaierror as e:
        return {
            "network": "OK",
            "dns": "FAILED",
            "https": "NOT TESTED",
            "bot_api": "NOT TESTED",
            "auth": "NOT TESTED",
            "category": "DNS_FAILURE",
            "bot_info": None,
            "reason": f"DNS resolution failed for {host}.",
            "detail": f"{host} could not be resolved. The bot token was NOT tested because the Telegram API could not be reached.",
        }
    except Exception as e:
        return {
            "network": "FAILED",
            "dns": "FAILED",
            "https": "NOT TESTED",
            "bot_api": "NOT TESTED",
            "auth": "NOT TESTED",
            "category": "NETWORK_FAILURE",
            "bot_info": None,
            "reason": f"Network resolution error for {host}: {e}",
            "detail": "The bot token was NOT tested because the Telegram API could not be reached.",
        }

    # 2. TCP connection
    try:
        s = socket.create_connection((host, port), timeout=timeout)
    except socket.timeout:
        return {
            "network": "OK",
            "dns": "OK",
            "https": "FAILED (TIMEOUT)",
            "bot_api": "NOT TESTED",
            "auth": "NOT TESTED",
            "category": "TIMEOUT",
            "bot_info": None,
            "reason": f"Connection to {host} timed out.",
            "detail": f"The bot token was NOT tested because TCP connection to port {port} timed out.",
        }
    except Exception as e:
        return {
            "network": "OK",
            "dns": "OK",
            "https": "FAILED",
            "bot_api": "NOT TESTED",
            "auth": "NOT TESTED",
            "category": "CONNECTION_FAILURE",
            "bot_info": None,
            "reason": f"Failed to connect to {host}:{port}: {e}",
            "detail": "The bot token was NOT tested because TCP connection failed.",
        }

    # 3. TLS / SSL handshake
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(s, server_hostname=host) as ss:
            pass
    except ssl.SSLError as e:
        return {
            "network": "OK",
            "dns": "OK",
            "https": "FAILED (TLS)",
            "bot_api": "NOT TESTED",
            "auth": "NOT TESTED",
            "category": "TLS_FAILURE",
            "bot_info": None,
            "reason": "TLS/SSL connection failed.",
            "detail": f"TLS handshake with {host} failed: {e}. The bot token was NOT tested.",
        }
    except Exception as e:
        return {
            "network": "OK",
            "dns": "OK",
            "https": "FAILED",
            "bot_api": "NOT TESTED",
            "auth": "NOT TESTED",
            "category": "TLS_FAILURE",
            "bot_info": None,
            "reason": f"SSL connection error: {e}",
            "detail": "The bot token was NOT tested.",
        }

    # 4. Telegram API getMe
    session = requests.Session()
    try:
        r = session.get(f"https://{host}/bot{token}/getMe", timeout=timeout)
    except requests.exceptions.Timeout:
        return {
            "network": "OK",
            "dns": "OK",
            "https": "OK",
            "bot_api": "FAILED (TIMEOUT)",
            "auth": "NOT TESTED",
            "category": "TIMEOUT",
            "bot_info": None,
            "reason": f"Connection to {host} timed out during API call.",
            "detail": "The bot token was NOT tested because the request timed out.",
        }
    except requests.exceptions.RequestException as e:
        return {
            "network": "OK",
            "dns": "OK",
            "https": "FAILED",
            "bot_api": "FAILED",
            "auth": "NOT TESTED",
            "category": "HTTP_FAILURE",
            "bot_info": None,
            "reason": f"HTTP request to {host} failed: {e}",
            "detail": "The bot token was NOT tested.",
        }

    # 5. Authentication result
    if r.status_code == 200:
        try:
            data = r.json()
            if data.get("ok"):
                bot_info = data.get("result", {})
                return {
                    "network": "OK",
                    "dns": "OK",
                    "https": "OK",
                    "bot_api": "OK",
                    "auth": "OK",
                    "category": "OK",
                    "bot_info": bot_info,
                    "reason": "Authenticated successfully.",
                    "detail": f"Connected as @{bot_info.get('username')}.",
                }
            else:
                return {
                    "network": "OK",
                    "dns": "OK",
                    "https": "OK",
                    "bot_api": "OK",
                    "auth": "FAILED",
                    "category": "AUTH_FAILURE",
                    "bot_info": None,
                    "reason": "Telegram rejected the bot token.",
                    "detail": data.get("description", "Unauthorized"),
                }
        except Exception as e:
            return {
                "network": "OK",
                "dns": "OK",
                "https": "OK",
                "bot_api": "FAILED",
                "auth": "FAILED",
                "category": "PARSE_ERROR",
                "bot_info": None,
                "reason": "Invalid JSON response from Telegram API.",
                "detail": str(e),
            }
    elif r.status_code in (401, 404):
        return {
            "network": "OK",
            "dns": "OK",
            "https": "OK",
            "bot_api": "OK",
            "auth": "FAILED",
            "category": "AUTH_FAILURE",
            "bot_info": None,
            "reason": "Telegram rejected the bot token.",
            "detail": f"HTTP {r.status_code}: Unauthorized or bot not found.",
        }
    else:
        return {
            "network": "OK",
            "dns": "OK",
            "https": "OK",
            "bot_api": f"FAILED (HTTP {r.status_code})",
            "auth": "FAILED",
            "category": "HTTP_ERROR",
            "bot_info": None,
            "reason": f"Telegram returned HTTP status {r.status_code}",
            "detail": r.text[:100],
        }


class TelegramClient:
    """Minimal, self-contained Telegram Bot API HTTP client."""

    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()
        self.last_diag: Optional[Dict[str, Any]] = None

    def check_connectivity(self) -> Dict[str, Any]:
        """Perform a full layered connectivity check."""
        self.last_diag = check_telegram_connectivity(self.token)
        return self.last_diag

    def get_me(self) -> Optional[Dict[str, Any]]:
        """Verify token and get bot metadata."""
        diag = self.check_connectivity()
        if diag.get("category") == "OK":
            return diag.get("bot_info")
        else:
            logger.warning(
                "Telegram getMe check failed [%s]: %s (%s)",
                diag.get("category"),
                diag.get("reason"),
                diag.get("detail", ""),
            )
            return None

    def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> list:
        """Fetch pending Telegram updates using long-polling."""
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset

        try:
            r = self.session.get(f"{self.base_url}/getUpdates", params=params, timeout=timeout + 10)
            data = r.json()
            if data.get("ok"):
                return data.get("result", [])
        except requests.exceptions.Timeout:
            return []
        except Exception as e:
            logger.warning("Error during getUpdates: %s", e)
            time.sleep(2)
        return []

    def send_message(self, chat_id: int, text: str, reply_markup: Optional[Dict] = None) -> Optional[int]:
        """Send a plain text message; returns the message_id or None."""
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            r = self.session.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=15,
            )
            data = r.json()
            if data.get("ok"):
                return data["result"]["message_id"]
        except Exception as e:
            logger.error("Failed to send Telegram message to chat %s: %s", chat_id, e)
        return None

    def edit_message_text(self, chat_id: int, message_id: int, text: str) -> bool:
        """Edit an existing message in place (used for live status updates)."""
        try:
            r = self.session.post(
                f"{self.base_url}/editMessageText",
                json={"chat_id": chat_id, "message_id": message_id, "text": text},
                timeout=10,
            )
            return r.json().get("ok", False)
        except Exception as e:
            logger.error("Failed to edit Telegram message: %s", e)
            return False

    def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        """Send a chat action (e.g. 'typing').

        Per official Telegram API docs the action expires after ~5 s,
        so callers must refresh it for long-running operations.
        https://core.telegram.org/bots/api#sendchataction
        """
        try:
            r = self.session.post(
                f"{self.base_url}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
                timeout=5,
            )
            return r.json().get("ok", False)
        except Exception as e:
            logger.warning("Failed to send chat action: %s", e)
            return False

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> bool:
        """Acknowledge a button press callback query."""
        try:
            r = self.session.post(
                f"{self.base_url}/answerCallbackQuery",
                json={"callback_query_id": callback_query_id, "text": text},
                timeout=5,
            )
            return r.json().get("ok", False)
        except Exception as e:
            logger.warning("Failed to answer callback query: %s", e)
            return False

    def send_photo(self, chat_id: int, photo_path: str, caption: str = "") -> bool:
        """Send a photo to a chat."""
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id, "caption": caption}
                r = self.session.post(
                    f"{self.base_url}/sendPhoto",
                    data=data,
                    files=files,
                    timeout=20,
                )
            return r.status_code == 200
        except Exception as e:
            logger.error("Failed to send photo to chat %s: %s", chat_id, e)
            return False


# ---------------------------------------------------------------------------
# Phase 16 — Typing keep-alive background thread
# ---------------------------------------------------------------------------

class _TypingKeepAlive:
    """Sends the Telegram 'typing' chat action every ~4 seconds while active.

    Usage:
        with _TypingKeepAlive(client, chat_id):
            # long-running work here
    """

    _INTERVAL = 4.0  # seconds between refreshes (Telegram action expires ~5 s)

    def __init__(self, client: TelegramClient, chat_id: int):
        self._client = client
        self._chat_id = chat_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        # Send immediately, then repeat until stopped
        while not self._stop.is_set():
            self._client.send_chat_action(self._chat_id, "typing")
            self._stop.wait(timeout=self._INTERVAL)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()


# ---------------------------------------------------------------------------
# Phase 17 — Inline keyboard for /controls
# ---------------------------------------------------------------------------

def _controls_keyboard() -> Dict:
    """Build the Telegram inline keyboard for manual robot control."""
    return {
        "inline_keyboard": [
            [{"text": "▲ UP",   "callback_data": "btn_up"}],
            [
                {"text": "◀ LEFT",  "callback_data": "btn_left"},
                {"text": "🏠 HOME", "callback_data": "btn_home"},
                {"text": "RIGHT ▶", "callback_data": "btn_right"},
            ],
            [{"text": "▼ DOWN", "callback_data": "btn_down"}],
            [
                {"text": "✊ CLOSE GRIPPER", "callback_data": "btn_close"},
                {"text": "✋ OPEN GRIPPER",  "callback_data": "btn_open"},
            ],
        ]
    }


def _handle_button(callback_data: str, flask_session: requests.Session) -> str:
    """Map a button callback to a Flask call and return a short result string.

    Buttons go directly to Flask, NOT through Qwen.
    RobotAPI safety validation applies as normal.
    """
    flask_base = FLASK_URL

    def _post(endpoint, body=None):
        try:
            r = flask_session.post(
                f"{flask_base}/{endpoint}",
                json=body or {},
                timeout=FLASK_TIMEOUT_SECONDS,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def _get(endpoint):
        try:
            r = flask_session.get(f"{flask_base}/{endpoint}", timeout=10)
            if r.status_code == 200:
                return r.json()
            return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    # Read current EEF position so we can compute relative moves
    def _current_eef():
        st = _get("state")
        if "error" in st:
            return None, st["error"]
        eef = st.get("eef", [0.0, -0.2, 0.1])
        return eef, None

    if callback_data == "btn_home":
        res = _post("reset_home")
        ok = res.get("success", False)
        return "🏠 Returned home." if ok else f"❌ Home failed: {res.get('error', {}).get('message', res)}"

    elif callback_data == "btn_open":
        res = _post("gripper", {"value": 0.0})
        ok = res.get("success", False)
        return "✋ Gripper opened." if ok else f"❌ Open failed: {res.get('error', {}).get('message', res)}"

    elif callback_data == "btn_close":
        res = _post("gripper", {"value": 1.0})
        ok = res.get("success", False)
        return "✊ Gripper closed." if ok else f"❌ Close failed: {res.get('error', {}).get('message', res)}"

    elif callback_data in ("btn_up", "btn_down", "btn_left", "btn_right"):
        eef, err = _current_eef()
        if err:
            return f"❌ Can't move: {err}"

        x, y, z = eef[0], eef[1], eef[2]
        labels = {
            "btn_up":    ("▲ UP",    (0,           0,           BUTTON_STEP_Z)),
            "btn_down":  ("▼ DOWN",  (0,           0,           -BUTTON_STEP_Z)),
            "btn_left":  ("◀ LEFT",  (-BUTTON_STEP_XY, 0,       0)),
            "btn_right": ("RIGHT ▶", (BUTTON_STEP_XY, 0,        0)),
        }
        label, (dx, dy, dz) = labels[callback_data]
        tx, ty, tz = round(x + dx, 4), round(y + dy, 4), round(z + dz, 4)

        res = _post("move_to", {"x": tx, "y": ty, "z": tz})
        ok = res.get("success", False)
        if ok:
            err_m = res.get("result", {}).get("position_error", "?")
            return f"{label}: moved to [{tx:.3f}, {ty:.3f}, {tz:.3f}] (err {err_m} m)"
        else:
            code = res.get("error", {}).get("code", "?")
            msg = res.get("error", {}).get("message", str(res))
            return f"❌ Move {label} failed ({code}): {msg}"

    return f"Unknown button: {callback_data}"


# ---------------------------------------------------------------------------
# TelegramBotRunner
# ---------------------------------------------------------------------------

class TelegramBotRunner:
    """Manages Telegram polling and per-chat serial work queues."""

    def __init__(self, token: str, agent: RobotArmAgent, event_bus: Optional[Any] = None):
        self.client = TelegramClient(token)
        self.agent = agent
        self.event_bus = event_bus
        self.running = False

        # Shared HTTP session for button Flask calls
        self._flask_session = requests.Session()

        # Per-chat serial queues: chat_id -> queue.Queue
        self.chat_queues: Dict[int, queue.Queue] = {}
        self.chat_workers: Dict[int, threading.Thread] = {}
        self.lock = threading.Lock()

    def _get_or_create_worker(self, chat_id: int) -> queue.Queue:
        """Ensure a dedicated serial queue and worker thread exist for this chat."""
        with self.lock:
            if chat_id not in self.chat_queues:
                q: queue.Queue = queue.Queue()
                self.chat_queues[chat_id] = q

                worker = threading.Thread(
                    target=self._chat_worker_loop,
                    args=(chat_id, q),
                    daemon=True,
                    name=f"Worker-Chat-{chat_id}",
                )
                self.chat_workers[chat_id] = worker
                worker.start()

            return self.chat_queues[chat_id]

    # -----------------------------------------------------------------------
    # Phase 16 — Status callback and status message management
    # -----------------------------------------------------------------------

    def _make_status_callback(
        self, chat_id: int, status_msg_ref: List[Optional[int]]
    ) -> Callable[[str, str, Any], None]:
        """Build a status callback that edits one Telegram message in place.

        status_msg_ref is a one-element list [message_id | None] shared
        between the callback and the caller so the ID can be set lazily
        on the first call.
        """
        action_labels = {
            "pick":       "Picking up the cube...",
            "place":      "Placing the cube...",
            "move_to":    "Moving to position...",
            "gripper":    "Adjusting gripper...",
            "reset_home": "Returning to home position...",
            "state":      "Checking state...",
            "camera":     "Capturing image...",
            "scenario":   "Running scenario...",
            "collisions": "Checking collisions...",
        }

        def callback(event: str, tool_name: str, data: Any):
            if self.event_bus:
                if event == "started":
                    self.event_bus.emit("tool_call", "agent", {
                        "name": tool_name,
                        "arguments": data if isinstance(data, dict) else {},
                        "chat_id": chat_id,
                    })
                elif event == "completed":
                    self.event_bus.emit("tool_result", "agent", {
                        "name": tool_name,
                        "success": True,
                        "detail": "",
                        "chat_id": chat_id,
                    })
                elif event == "failed":
                    err_info = ""
                    if isinstance(data, dict):
                        err = data.get("error") or {}
                        if isinstance(err, dict):
                            err_info = err.get("message", "")
                    self.event_bus.emit("tool_result", "agent", {
                        "name": tool_name,
                        "success": False,
                        "detail": err_info,
                        "chat_id": chat_id,
                    })

            if event == "started":
                label = action_labels.get(tool_name, f"Executing {tool_name}...")
                if status_msg_ref[0] is None:
                    # Send the initial status message
                    mid = self.client.send_message(chat_id, f"⏳ {label}")
                    status_msg_ref[0] = mid
                else:
                    self.client.edit_message_text(chat_id, status_msg_ref[0], f"⏳ {label}")

            elif event == "completed":
                if status_msg_ref[0] is not None:
                    self.client.edit_message_text(chat_id, status_msg_ref[0], f"✅ Done.")

            elif event == "failed":
                err_info = ""
                if isinstance(data, dict):
                    err = data.get("error") or {}
                    if isinstance(err, dict):
                        err_info = f": {err.get('message', '')}"
                if status_msg_ref[0] is not None:
                    self.client.edit_message_text(
                        chat_id, status_msg_ref[0], f"❌ Action failed{err_info}"
                    )

        return callback

    # -----------------------------------------------------------------------
    # Worker loop (per chat)
    # -----------------------------------------------------------------------

    def _chat_worker_loop(self, chat_id: int, q: queue.Queue):
        """Processes messages sequentially for a single chat."""
        logger.info("Started serial worker for chat %s", chat_id)
        while self.running:
            try:
                task = q.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:   # Shutdown sentinel
                break

            task_type = task.get("type")

            # ---------------------------------------------------------------
            # Phase 11 — /new: handled entirely outside the LLM
            # ---------------------------------------------------------------
            if task_type == "new_conversation":
                self.agent.reset_conversation(chat_id)
                self.client.send_message(chat_id, "Conversation reset. 🔄")
                q.task_done()
                continue

            # ---------------------------------------------------------------
            # Phase 17 — Button callback: goes directly to Flask, not Qwen
            # ---------------------------------------------------------------
            if task_type == "button":
                callback_data = task.get("callback_data", "")
                callback_query_id = task.get("callback_query_id", "")
                try:
                    result_text = _handle_button(callback_data, self._flask_session)
                    self.client.answer_callback_query(callback_query_id, text=result_text[:200])
                    self.client.send_message(chat_id, result_text)
                except Exception as e:
                    logger.error("Button handler error: %s", e)
                    self.client.answer_callback_query(callback_query_id, text="Error")
                    self.client.send_message(chat_id, f"Button error: {e}")
                q.task_done()
                continue

            # ---------------------------------------------------------------
            # Normal agent message
            # ---------------------------------------------------------------
            if task_type == "message":
                user_text = task.get("text", "")
                logger.info("Processing message for chat %s: '%s'", chat_id, user_text)

                if self.event_bus:
                    self.event_bus.emit("telegram_in", "telegram", {
                        "text": user_text,
                        "chat_id": chat_id,
                    })
                    self.event_bus.emit("agent_start", "telegram", {
                        "text": user_text,
                        "chat_id": chat_id,
                    })

                # Phase 17 — /controls command
                if user_text.strip().lower() == "/controls":
                    self.client.send_message(
                        chat_id,
                        "🤖 Manual Robot Controls\nButtons call the robot directly — no AI involved.",
                        reply_markup=_controls_keyboard(),
                    )
                    if self.event_bus:
                        self.event_bus.emit("agent_end", "telegram", {"chat_id": chat_id})
                    q.task_done()
                    continue

                # Phase 16 — typing keep-alive + status message
                status_msg_ref: List[Optional[int]] = [None]
                status_cb = self._make_status_callback(chat_id, status_msg_ref)

                try:
                    with _TypingKeepAlive(self.client, chat_id):
                        result = self.agent.process_message(
                            user_text=user_text,
                            chat_id=chat_id,
                            status_callback=status_cb,
                        )

                    reply_text = result.get("reply", "Done.")
                    photos = result.get("photos", [])

                    if self.event_bus:
                        self.event_bus.emit("telegram_out", "agent", {
                            "reply": reply_text,
                            "photos": photos,
                            "chat_id": chat_id,
                        })

                    # Clean up status message if it exists (final reply replaces it)
                    if status_msg_ref[0] is not None:
                        # Delete the interim status message and send the real reply
                        # (Telegram doesn't have delete in basic API without admin rights;
                        # edit it to empty or just send the reply as a new message)
                        self.client.edit_message_text(
                            chat_id, status_msg_ref[0], f"💬 {reply_text}"
                        )
                    else:
                        if reply_text:
                            self.client.send_message(chat_id, reply_text)

                    # Send photos if any camera shots were taken
                    for photo_path in photos:
                        if os.path.exists(photo_path):
                            self.client.send_photo(chat_id, photo_path, caption="Camera snapshot")

                except Exception as e:
                    logger.error("Unhandled error processing chat message: %s", e, exc_info=True)
                    if self.event_bus:
                        self.event_bus.emit("error", "telegram", {
                            "msg": str(e),
                            "chat_id": chat_id,
                        })
                    self.client.send_message(
                        chat_id,
                        f"An error occurred while handling your request: {e}",
                    )
                finally:
                    if self.event_bus:
                        self.event_bus.emit("agent_end", "telegram", {
                            "chat_id": chat_id,
                        })
                    q.task_done()

    def start(self):
        """Start long-polling loop."""
        bot_info = self.client.get_me()
        if not bot_info:
            diag = getattr(self.client, "last_diag", None) or {}
            reason = diag.get("reason", "authentication failed")
            logger.warning("Telegram polling stopped: %s", reason)
            self.running = False
            return

        username = bot_info.get("username", "RobotArmBot")
        logger.info("Bot authenticated as @%s. Starting long-polling...", username)

        self.running = True
        offset = None

        try:
            while self.running:
                updates = self.client.get_updates(offset=offset, timeout=10)
                for update in updates:
                    offset = update["update_id"] + 1

                    # -------------------------------------------------------
                    # Phase 17 — Handle inline button callbacks
                    # -------------------------------------------------------
                    if "callback_query" in update:
                        cq = update["callback_query"]
                        cq_id = cq["id"]
                        cq_data = cq.get("data", "")
                        cq_chat_id = cq["message"]["chat"]["id"]
                        q = self._get_or_create_worker(cq_chat_id)
                        q.put({
                            "type": "button",
                            "callback_data": cq_data,
                            "callback_query_id": cq_id,
                        })
                        continue

                    msg = update.get("message")
                    if not msg or "text" not in msg:
                        continue

                    chat_id = msg["chat"]["id"]
                    user_text = msg["text"].strip()

                    # Phase 11 — intercept /new before it reaches the agent
                    if user_text.lower() in ("/new", "/new@" + username.lower()):
                        q = self._get_or_create_worker(chat_id)
                        q.put({"type": "new_conversation"})
                        continue

                    # Normal message
                    q = self._get_or_create_worker(chat_id)
                    q.put({"type": "message", "text": user_text})

        except KeyboardInterrupt:
            logger.info("Stopping bot on KeyboardInterrupt...")
        finally:
            self.stop()

    def stop(self):
        """Stop polling and cleanly terminate workers."""
        self.running = False
        with self.lock:
            for q in self.chat_queues.values():
                q.put(None)
        logger.info("Bot runner shut down successfully.")


# ---------------------------------------------------------------------------
# CLI fallback mode
# ---------------------------------------------------------------------------

def run_cli_mode(agent: RobotArmAgent):
    """Interactive CLI mode for direct terminal testing without Telegram."""
    print("\n" + "=" * 60)
    print("  MINI OPENCLAW — ROBOT ARM AGENT (CLI TEST MODE)")
    print("  Type your command or prompt.")
    print("  Type '/new' to reset conversation.")
    print("  Type 'exit' or 'quit' to end.")
    print("=" * 60 + "\n")

    chat_id = 999
    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI mode.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting CLI mode.")
            break

        # Phase 11 — /new in CLI mode
        if user_input.lower() == "/new":
            agent.reset_conversation(chat_id)
            print("Agent > Conversation reset.")
            print("-" * 60)
            continue

        print("\n[Agent thinking & executing...]")

        def cli_status_callback(event: str, tool_name: str, data: Any):
            """Print status events to the terminal in CLI mode."""
            action_labels = {
                "pick": "Picking up the cube",
                "place": "Placing the cube",
                "move_to": "Moving to position",
                "gripper": "Adjusting gripper",
                "reset_home": "Returning to home",
                "state": "Checking state",
            }
            label = action_labels.get(tool_name, tool_name)
            if event == "started":
                print(f"  [STATUS] {label}...")
            elif event == "completed":
                print(f"  [STATUS] {label} — done.")
            elif event == "failed":
                print(f"  [STATUS] {label} — FAILED.")

        res = agent.process_message(user_input, chat_id=chat_id, status_callback=cli_status_callback)

        print(f"\nAgent > {res['reply']}")
        print(f"  Rounds: {res.get('agent_rounds', '?')} | Physical calls: {res.get('physical_tool_calls', '?')}")
        if res.get("failure_type"):
            print(f"  Failure type: {res['failure_type']}")
        if res.get("tool_calls"):
            print("  Tool calls:")
            for tc in res["tool_calls"]:
                success_str = "✓" if tc.get("tool_success") else "✗"
                print(f"    {success_str} {tc['name']}({tc['arguments']}) valid={tc['valid']}")
        if res.get("photos"):
            print(f"  Photos captured: {res['photos']}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mini Openclaw: Telegram-to-Robot-Arm Agent")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal CLI mode instead of Telegram")
    args = parser.parse_args()

    agent = RobotArmAgent()

    if args.cli or not TELEGRAM_BOT_TOKEN:
        if not TELEGRAM_BOT_TOKEN and not args.cli:
            print("[INFO] No TELEGRAM_BOT_TOKEN found in environment or .env file.")
            print("[INFO] Falling back to interactive CLI test mode.\n")
        run_cli_mode(agent)
    else:
        runner = TelegramBotRunner(TELEGRAM_BOT_TOKEN, agent)
        runner.start()


if __name__ == "__main__":
    main()
