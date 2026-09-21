"""
bot.py — Lightweight Telegram Bot Runner for the SO-100 Robot Arm Agent.

Features:
- Direct Telegram Bot API long-polling (zero external framework overhead)
- Per-chat serial queue ensuring sequential execution of commands
- Sends replies and camera photos
- Graceful error boundary: never crashes on network drops or model issues
- Interactive terminal fallback mode for instant local testing without Telegram
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, Optional

import requests

from agent import RobotArmAgent
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TelegramArmBot")


class TelegramClient:
    """Minimal, self-contained Telegram Bot API HTTP client."""

    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()

    def get_me(self) -> Optional[Dict[str, Any]]:
        """Verify token and get bot metadata."""
        try:
            r = self.session.get(f"{self.base_url}/getMe", timeout=10)
            data = r.json()
            if data.get("ok"):
                return data.get("result")
        except Exception as e:
            logger.error("Failed to connect to Telegram getMe: %s", e)
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

    def send_message(self, chat_id: int, text: str) -> bool:
        """Send a plain text message to a chat."""
        try:
            r = self.session.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=15
            )
            return r.status_code == 200
        except Exception as e:
            logger.error("Failed to send Telegram message to chat %s: %s", chat_id, e)
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
                    timeout=20
                )
            return r.status_code == 200
        except Exception as e:
            logger.error("Failed to send photo to chat %s: %s", chat_id, e)
            return False


class TelegramBotRunner:
    """Manages Telegram polling and per-chat serial work queues."""

    def __init__(self, token: str, agent: RobotArmAgent):
        self.client = TelegramClient(token)
        self.agent = agent
        self.running = False

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
                    name=f"Worker-Chat-{chat_id}"
                )
                self.chat_workers[chat_id] = worker
                worker.start()

            return self.chat_queues[chat_id]

    def _chat_worker_loop(self, chat_id: int, q: queue.Queue):
        """Processes messages sequentially for a single chat."""
        logger.info("Started serial worker for chat %s", chat_id)
        while self.running:
            try:
                task = q.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # Shutdown sentinel
                break

            user_text, msg_id = task
            logger.info("Processing message for chat %s: '%s'", chat_id, user_text)

            try:
                result = self.agent.process_message(user_text=user_text, chat_id=chat_id)
                reply_text = result.get("reply", "Done.")
                photos = result.get("photos", [])

                # Send text response
                if reply_text:
                    self.client.send_message(chat_id, reply_text)

                # Send photos if any camera shots were taken
                for photo_path in photos:
                    if os.path.exists(photo_path):
                        self.client.send_photo(chat_id, photo_path, caption="Camera snapshot")

            except Exception as e:
                logger.error("Unhandled error processing chat message: %s", e, exc_info=True)
                self.client.send_message(
                    chat_id,
                    f"An error occurred while handling your request: {e}"
                )
            finally:
                q.task_done()

        logger.info("Exiting worker for chat %s", chat_id)

    def start(self):
        """Start long-polling loop."""
        bot_info = self.client.get_me()
        if not bot_info:
            logger.error("Invalid bot token or Telegram unreachable. Exiting.")
            sys.exit(1)

        username = bot_info.get("username", "RobotArmBot")
        logger.info("Bot authenticated as @%s. Starting long-polling...", username)

        self.running = True
        offset = None

        try:
            while self.running:
                updates = self.client.get_updates(offset=offset, timeout=20)
                for update in updates:
                    offset = update["update_id"] + 1
                    msg = update.get("message")
                    if not msg or "text" not in msg:
                        continue

                    chat_id = msg["chat"]["id"]
                    user_text = msg["text"].strip()
                    msg_id = msg["message_id"]

                    # Enqueue to per-chat serial queue
                    q = self._get_or_create_worker(chat_id)
                    q.put((user_text, msg_id))

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


def run_cli_mode(agent: RobotArmAgent):
    """Interactive CLI mode for direct terminal testing without Telegram."""
    print("\n" + "=" * 60)
    print("  MINI OPENCLAW — ROBOT ARM AGENT (CLI TEST MODE)")
    print("  Type your command or prompt. Type 'exit' or 'quit' to end.")
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

        print("\n[Agent thinking & executing...]")
        res = agent.process_message(user_input, chat_id=chat_id)

        print(f"\nAgent > {res['reply']}")
        if res.get("tool_calls"):
            print("Tool calls issued:")
            for tc in res["tool_calls"]:
                print(f"  • {tc['name']}({tc['arguments']}) -> valid={tc['valid']}")
        if res.get("photos"):
            print(f"Photos captured: {res['photos']}")
        print("-" * 60)


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
