"""Mocked Telegram transport and runner reliability regression tests."""

import threading
import time
from unittest.mock import MagicMock, patch

import requests

from bot import TelegramBotRunner, TelegramClient, TelegramRequestFailure


def _response(status=200, body=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body if body is not None else {"ok": True, "result": []}
    return response


def test_get_updates_retries_connection_failure_then_succeeds():
    client = TelegramClient("test-token")
    client.session.request = MagicMock(side_effect=[
        requests.exceptions.ConnectionError("connection reset"),
        _response(body={"ok": True, "result": [{"update_id": 7}]}),
    ])

    with patch("bot.time.sleep"):
        assert client.get_updates(timeout=1) == [{"update_id": 7}]

    assert client.session.request.call_count == 2
    assert client.last_failure is None


def test_get_updates_stops_after_bounded_repeated_transient_failures():
    client = TelegramClient("test-token")
    client.session.request = MagicMock(
        side_effect=requests.exceptions.ConnectionError("connection reset")
    )

    with patch("bot.time.sleep"):
        assert client.get_updates(timeout=1) is None

    assert client.session.request.call_count == 3
    assert client.last_failure.category == "NETWORK"


def test_send_message_retries_dns_failure_but_not_ambiguous_connection_failure():
    dns_client = TelegramClient("test-token")
    dns_client.session.request = MagicMock(side_effect=[
        requests.exceptions.ConnectionError("NameResolutionError: getaddrinfo failed"),
        _response(body={"ok": True, "result": {"message_id": 42}}),
    ])
    with patch("bot.time.sleep"):
        assert dns_client.send_message(1, "hello") == 42
    assert dns_client.session.request.call_count == 2

    ambiguous_client = TelegramClient("test-token")
    ambiguous_client.session.request = MagicMock(
        side_effect=requests.exceptions.ConnectionError("connection reset")
    )
    assert ambiguous_client.send_message(1, "hello") is None
    # A reset can occur after Telegram accepted the POST, so it is not retried.
    assert ambiguous_client.session.request.call_count == 1


def test_timeout_retries_for_idempotent_chat_action_and_auth_does_not():
    client = TelegramClient("test-token")
    client.session.request = MagicMock(side_effect=[
        requests.exceptions.Timeout("timed out"),
        _response(body={"ok": True}),
    ])
    with patch("bot.time.sleep"):
        assert client.send_chat_action(1) is True
    assert client.session.request.call_count == 2

    client.session.request = MagicMock(return_value=_response(status=401, body={"ok": False}))
    assert client.send_chat_action(1) is False
    assert client.session.request.call_count == 1
    assert client.last_failure.category == "AUTH"


def test_non_retryable_http_error_fails_once():
    client = TelegramClient("test-token")
    client.session.request = MagicMock(return_value=_response(status=400, body={"ok": False}))

    assert client.edit_message_text(1, 2, "status") is False
    assert client.session.request.call_count == 1
    assert client.last_failure.category == "API"
    assert client.last_failure.transient is False


def test_runner_marks_offline_then_reconnects_in_same_polling_loop():
    runner = TelegramBotRunner("test-token", MagicMock())
    runner.client.get_me = MagicMock(return_value={"username": "testbot"})
    calls = []

    def poll(**_kwargs):
        calls.append(1)
        if len(calls) == 1:
            runner.client.last_failure = TelegramRequestFailure("DNS", "DNS failed", True)
            return None
        if len(calls) == 3:
            runner.stop()
        return []

    runner.client.get_updates = MagicMock(side_effect=poll)
    runner.start()

    assert len(calls) == 3
    assert runner.polling_online is False  # stop() clears it after reconnect.
    assert runner._offline is False


def test_runner_shutdown_interrupts_reconnect_wait():
    runner = TelegramBotRunner("test-token", MagicMock())
    runner.client.get_me = MagicMock(return_value={"username": "testbot"})

    def poll(**_kwargs):
        runner.client.last_failure = TelegramRequestFailure("NETWORK", "offline", True)
        return None

    runner.client.get_updates = MagicMock(side_effect=poll)
    thread = threading.Thread(target=runner.start)
    thread.start()
    deadline = time.time() + 1
    while runner._poll_failures == 0 and time.time() < deadline:
        time.sleep(0.01)
    runner.stop()
    thread.join(timeout=1)

    assert not thread.is_alive()


def test_runner_stop_is_idempotent():
    runner = TelegramBotRunner("test-token", MagicMock())
    runner.running = True

    with patch("bot.logger.info") as info:
        runner.stop()
        runner.stop()

    assert info.call_count == 1


def test_delivery_failure_never_recomputes_agent_result():
    processed = threading.Event()

    class Agent:
        calls = 0

        def process_message(self, **_kwargs):
            self.calls += 1
            processed.set()
            return {"reply": "robot action already completed", "photos": []}

    agent = Agent()
    runner = TelegramBotRunner("test-token", agent)
    runner.running = True
    runner.client.send_chat_action = MagicMock(return_value=False)
    runner.client.send_message = MagicMock(return_value=None)  # final delivery fails

    q = runner._get_or_create_worker(101)
    q.put({"type": "message", "text": "move"})
    assert processed.wait(1)
    q.join()
    runner.stop()

    assert agent.calls == 1
    assert runner.client.send_message.call_count == 1


def test_per_chat_queue_keeps_agent_execution_serial():
    finished = threading.Event()

    class Agent:
        active = 0
        max_active = 0
        calls = 0
        lock = threading.Lock()

        def process_message(self, **_kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
                self.calls += 1
                if self.calls == 2:
                    finished.set()
            return {"reply": "done", "photos": []}

    agent = Agent()
    runner = TelegramBotRunner("test-token", agent)
    runner.running = True
    runner.client.send_chat_action = MagicMock(return_value=True)
    runner.client.send_message = MagicMock(return_value=1)
    q = runner._get_or_create_worker(202)
    q.put({"type": "message", "text": "first"})
    q.put({"type": "message", "text": "second"})
    assert finished.wait(2)
    q.join()
    runner.stop()

    assert agent.max_active == 1
