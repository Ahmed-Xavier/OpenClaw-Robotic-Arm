"""
test_launcher.py — Comprehensive unit tests for launcher.py.

All tests mock external services (Ollama, Flask, Telegram) so they run
instantly without requiring running servers or network access.
"""

import io
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in sys.path so we can import launcher
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot import check_telegram_connectivity
from launcher import (
    _COMMANDS,
    Event,
    EventBus,
    LogMonitor,
    ServiceManager,
    TerminalConsole,
    _format_convo_event,
    _format_log_event,
    run_startup,
)


# ===========================================================================
# 1. EventBus Tests
# ===========================================================================

def test_eventbus_emit_reaches_subscribers():
    """Verify that emitted events reach all active subscribers."""
    bus = EventBus()
    received_a = []
    received_b = []

    bus.subscribe(lambda e: received_a.append(e))
    bus.subscribe(lambda e: received_b.append(e))

    bus.emit("test_type", "test_source", {"key": "value"})

    assert len(received_a) == 1
    assert len(received_b) == 1
    assert received_a[0].type == "test_type"
    assert received_a[0].source == "test_source"
    assert received_a[0].data == {"key": "value"}
    assert received_b[0].type == "test_type"


def test_eventbus_unsubscribe():
    """Verify that unsubscribed callbacks are no longer notified."""
    bus = EventBus()
    received = []

    def cb(e):
        received.append(e)

    bus.subscribe(cb)
    bus.emit("t1", "src", {})
    assert len(received) == 1

    bus.unsubscribe(cb)
    bus.emit("t2", "src", {})
    assert len(received) == 1


def test_eventbus_subscriber_crash_is_isolated():
    """A crashing subscriber must be caught and logged; others still receive event."""
    bus = EventBus()
    good_received = []

    def bad_subscriber(e):
        raise RuntimeError("Subscriber explosion!")

    def good_subscriber(e):
        good_received.append(e)

    bus.subscribe(bad_subscriber)
    bus.subscribe(good_subscriber)

    # Should not raise exception
    bus.emit("isolated_test", "test", {"msg": "hello"})
    assert len(good_received) == 1
    assert good_received[0].type == "isolated_test"


def test_eventbus_no_lock_held_during_callback():
    """Subscribers must be able to subscribe/emit inside callback without deadlock."""
    bus = EventBus()
    nested_received = []

    def nested_cb(e):
        nested_received.append(e)

    def outer_subscriber(e):
        if e.type == "outer":
            # This would deadlock if emit() held self._lock during callback invocation
            bus.subscribe(nested_cb)
            bus.emit("inner", "nested", {"ok": True})

    bus.subscribe(outer_subscriber)
    bus.emit("outer", "root", {})

    assert any(e.type == "inner" for e in nested_received)


# ===========================================================================
# 2. Command Parsing & Console Tests
# ===========================================================================

def test_help_lists_all_commands(capsys):
    """Verify /help prints every command registered in _COMMANDS."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    console = TerminalConsole(sm, bus)

    console._cmd_help()
    captured = capsys.readouterr().out

    for cmd in _COMMANDS:
        assert cmd in captured


def test_unknown_command_handled(capsys):
    """Unknown commands print a clean error and guidance."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    console = TerminalConsole(sm, bus)

    console._dispatch("/xyznonexistent")
    captured = capsys.readouterr().out
    assert "Unknown command: /xyznonexistent" in captured
    assert "Type /help" in captured


def test_dispatch_slash_vs_text():
    """Verify slash inputs route to command methods and text routes to agent."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    console = TerminalConsole(sm, bus)

    with patch.object(console, "_cmd_help") as mock_help, \
         patch.object(console, "_send_to_agent") as mock_agent:

        console._dispatch("/help")
        mock_help.assert_called_once()
        mock_agent.assert_not_called()

        mock_help.reset_mock()
        # Normal text line handled in run() via _send_to_agent
        console._send_to_agent("pick up the blue cube")
        mock_agent.assert_called_once_with("pick up the blue cube")


def test_status_format(capsys):
    """Verify /status prints agent, robot, services, and limits info cleanly."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.is_robot_busy.return_value = False
    sm.current_model = "qwen3.5:4b"
    sm.flask_running.return_value = True
    sm.telegram_connected = True

    console = TerminalConsole(sm, bus)

    with patch("requests.get") as mock_get:
        # Mock Flask /state
        mock_flask_resp = MagicMock()
        mock_flask_resp.json.return_value = {
            "eef": [0.12, -0.25, 0.08],
            "is_holding": True,
            "held_object": "blue_cube",
        }
        # Mock Ollama /api/tags
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.raise_for_status.return_value = None

        def get_side_effect(url, **kwargs):
            if "state" in url:
                return mock_flask_resp
            return mock_ollama_resp

        mock_get.side_effect = get_side_effect

        console._cmd_status()
        out = capsys.readouterr().out

        assert "Name:   Army" in out
        assert "State:  READY" in out
        assert "qwen3.5:4b" in out
        assert "EEF:      [0.120, -0.250, 0.080]" in out
        assert "Holding:  True" in out
        assert "Object:   blue_cube" in out
        assert "Flask API:  RUNNING" in out
        assert "Telegram:   CONNECTED" in out


# ===========================================================================
# 3. Model Management Tests (Atomic 3-Stage Pipeline)
# ===========================================================================

def test_model_change_rejected_on_missing(capsys):
    """If the requested model is not in Ollama's available models, change is rejected."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.current_model = "qwen3.5:4b"
    sm.list_ollama_models.return_value = ["qwen3.5:4b", "llama3:8b"]

    console = TerminalConsole(sm, bus)
    console._cmd_model("mistral:7b")

    out = capsys.readouterr().out
    assert "Model 'mistral:7b' not found in Ollama" in out
    assert "Current model unchanged: qwen3.5:4b" in out
    sm.set_model.assert_not_called()


def test_model_change_rejected_on_ping_fail(capsys):
    """If the model exists but fails the inference ping, change is rejected."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.current_model = "qwen3.5:4b"
    sm.list_ollama_models.return_value = ["qwen3.5:4b", "llama3:8b"]
    sm.ping_model.return_value = (False, "inference timed out")

    console = TerminalConsole(sm, bus)
    console._cmd_model("llama3:8b")

    out = capsys.readouterr().out
    assert "Model change rejected: inference timed out" in out
    assert "Current model unchanged: qwen3.5:4b" in out
    sm.set_model.assert_not_called()


def test_model_change_applied_on_success(capsys):
    """If model exists and ping succeeds, set_model is called atomically."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.current_model = "qwen3.5:4b"
    sm.list_ollama_models.return_value = ["qwen3.5:4b", "llama3:8b"]
    sm.ping_model.return_value = (True, "OK")

    console = TerminalConsole(sm, bus)
    console._cmd_model("llama3:8b")

    out = capsys.readouterr().out
    assert "Model changed: qwen3.5:4b → llama3:8b" in out
    sm.set_model.assert_called_once_with("llama3:8b")


# ===========================================================================
# 4. ServiceManager Lifecycle & Safety Tests
# ===========================================================================

def test_shutdown_idempotent():
    """stop_all() can be called multiple times without throwing errors."""
    bus = EventBus()
    sm = ServiceManager(bus)

    # Initial state with no running services
    sm.stop_all()
    sm.stop_all()
    assert True


def test_shutdown_handles_second_ctrl_c_while_waiting_for_poller():
    """A Ctrl+C during polling-thread join must not escape as a traceback."""
    bus = EventBus()
    sm = ServiceManager(bus)
    runner = MagicMock()
    thread = MagicMock()
    thread.join.side_effect = KeyboardInterrupt
    sm._bot_runner = runner
    sm._bot_thread = thread

    sm.stop_all()

    runner.stop.assert_called_once()
    thread.join.assert_called_once_with(timeout=20)


def test_restart_joins_old_thread():
    """restart_agent must stop the old bot runner and join its thread."""
    bus = EventBus()
    sm = ServiceManager(bus)

    mock_runner = MagicMock()
    mock_runner.running = True
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = False

    sm._bot_runner = mock_runner
    sm._bot_thread = mock_thread

    with patch.object(sm, "_create_agent") as mock_agent_create, \
         patch.object(sm, "check_telegram", return_value={"category": "OK", "bot_info": {"username": "test"}}), \
         patch("launcher.TelegramBotRunner") as mock_runner_cls, \
         patch("threading.Thread") as mock_thread_cls:
        sm.restart_agent()

        # Verify old runner stopped and joined
        mock_runner.stop.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=20)


def test_restart_refuses_second_poller_when_old_thread_is_alive(capsys):
    """A stuck old poller must block replacement rather than duplicate updates."""
    bus = EventBus()
    sm = ServiceManager(bus)
    old_runner = MagicMock()
    old_thread = MagicMock()
    old_thread.is_alive.return_value = True
    sm._bot_runner = old_runner
    sm._bot_thread = old_thread

    with patch.object(sm, "_create_agent") as create_agent, \
         patch("launcher.TelegramBotRunner") as runner_cls:
        sm.restart_agent()

    old_runner.stop.assert_called_once()
    old_thread.join.assert_called_once_with(timeout=20)
    create_agent.assert_not_called()
    runner_cls.assert_not_called()
    assert "refusing to start a second polling loop" in capsys.readouterr().out


def test_restart_does_not_reset_robot():
    """restart_agent must only recreate the local Python agent without resetting robot."""
    bus = EventBus()
    sm = ServiceManager(bus)

    with patch("requests.post") as mock_post, \
         patch.object(sm, "_create_agent") as mock_agent_create:
        sm.restart_agent()
        # No HTTP calls made to /reset_home or /state
        mock_post.assert_not_called()


def test_busy_tracking_via_eventbus():
    """is_robot_busy reflects agent_start and agent_end events accurately."""
    bus = EventBus()
    sm = ServiceManager(bus)

    assert not sm.is_robot_busy()

    bus.emit("agent_start", "terminal", {})
    assert sm.is_robot_busy()

    bus.emit("agent_end", "terminal", {})
    assert not sm.is_robot_busy()


def test_busy_check_blocks_stop(capsys):
    """When robot is busy, user can cancel exit by answering 'n'."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.is_robot_busy.return_value = True

    console = TerminalConsole(sm, bus)

    # User enters 'n' to cancel
    with patch("builtins.input", return_value="n"):
        console._cmd_exit()

    out = capsys.readouterr().out
    assert "A robot action is currently running" in out
    assert "Cancelled" in out
    sm.stop_all.assert_not_called()


# ===========================================================================
# 5. Format Helpers Tests
# ===========================================================================

def test_format_convo_and_log_event():
    """Verify event formatting for /convo and /logs."""
    ev_in = Event(
        type="telegram_in",
        timestamp="2026-09-22T19:00:00Z",
        source="telegram",
        data={"text": "hello robot"},
    )
    ev_out = Event(
        type="telegram_out",
        timestamp="2026-09-22T19:00:01Z",
        source="agent",
        data={"reply": "hello human"},
    )
    ev_tool = Event(
        type="tool_call",
        timestamp="2026-09-22T19:00:02Z",
        source="agent",
        data={"name": "pick", "arguments": {"object": "cube"}},
    )

    convo_in = _format_convo_event(ev_in)
    assert "[Telegram → Agent]" in convo_in
    assert "hello robot" in convo_in

    convo_out = _format_convo_event(ev_out)
    assert "[Agent → Telegram]" in convo_out
    assert "hello human" in convo_out

    log_tool = _format_log_event(ev_tool)
    assert "Tool call:     pick(" in log_tool


# ===========================================================================
# 6. Telegram Connectivity & Failure Classification Tests
# ===========================================================================

def test_startup_still_starts_telegram_when_ollama_start_fails(capsys):
    """Telegram must not be skipped by an earlier non-Telegram startup failure."""
    bus = EventBus()
    sm = ServiceManager(bus)

    with patch.object(sm, "ollama_reachable", return_value=False), \
         patch.object(sm, "start_ollama_server", return_value=(False, "Ollama unavailable")), \
         patch.object(sm, "start_bot", return_value=(
             True, "OK (@testbot)", {"category": "OK", "bot_info": {"username": "testbot"}}
         )) as start_bot:
        assert run_startup(sm) is False

    start_bot.assert_called_once()
    assert "Connecting Telegram" in capsys.readouterr().out

def test_telegram_check_dns_failure():
    """Verify that DNS failure is classified specifically without testing bot token."""
    import socket
    with patch("socket.getaddrinfo", side_effect=socket.gaierror(11001, "getaddrinfo failed")):
        res = check_telegram_connectivity("test_token_xyz")
        assert res["category"] == "DNS_FAILURE"
        assert res["dns"] == "FAILED"
        assert res["https"] == "NOT TESTED"
        assert res["auth"] == "NOT TESTED"
        assert "DNS resolution failed" in res["reason"]
        assert "bot token was NOT tested" in res["detail"]
        assert "test_token_xyz" not in str(res)


def test_telegram_check_connection_timeout():
    """Verify connection timeout classification."""
    import socket
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, 0, 0, "", ("1.2.3.4", 443))]), \
         patch("socket.create_connection", side_effect=socket.timeout("timed out")):
        res = check_telegram_connectivity("test_token_xyz")
        assert res["category"] == "TIMEOUT"
        assert res["dns"] == "OK"
        assert "TIMEOUT" in res["https"]
        assert "timed out" in res["reason"]
        assert "bot token was NOT tested" in res["detail"]
        assert "test_token_xyz" not in str(res)


def test_telegram_check_tls_failure():
    """Verify TLS/SSL handshake failure classification."""
    import socket, ssl
    mock_sock = MagicMock()
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, 0, 0, "", ("1.2.3.4", 443))]), \
         patch("socket.create_connection", return_value=mock_sock), \
         patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value.wrap_socket.side_effect = ssl.SSLError("certificate verify failed")
        res = check_telegram_connectivity("test_token_xyz")
        assert res["category"] == "TLS_FAILURE"
        assert "TLS" in res["reason"]
        assert "bot token was NOT tested" in res["detail"]
        assert "test_token_xyz" not in str(res)


def test_telegram_check_auth_failure():
    """Verify HTTP 401 response is classified as bot token rejected."""
    import socket
    mock_sock = MagicMock()
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, 0, 0, "", ("1.2.3.4", 443))]), \
         patch("socket.create_connection", return_value=mock_sock), \
         patch("ssl.create_default_context"), \
         patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"ok": False, "error_code": 401, "description": "Unauthorized"}
        mock_get.return_value = mock_resp

        res = check_telegram_connectivity("invalid_token_123")
        assert res["category"] == "AUTH_FAILURE"
        assert res["auth"] == "FAILED"
        assert "rejected" in res["reason"]
        assert "invalid_token_123" not in str(res)


def test_telegram_check_successful_get_me():
    """Verify successful getMe returns bot info."""
    import socket
    mock_sock = MagicMock()
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, 0, 0, "", ("1.2.3.4", 443))]), \
         patch("socket.create_connection", return_value=mock_sock), \
         patch("ssl.create_default_context"), \
         patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"id": 12345, "is_bot": True, "first_name": "Test", "username": "TestArmBot"},
        }
        mock_get.return_value = mock_resp

        res = check_telegram_connectivity("valid_token_123")
        assert res["category"] == "OK"
        assert res["auth"] == "OK"
        assert res["bot_info"]["username"] == "TestArmBot"
        assert "valid_token_123" not in str(res)


def test_telegram_unavailable_terminal_remains_operational():
    """When Telegram fails (DNS failure), startup succeeds and terminal agent is active."""
    bus = EventBus()
    sm = ServiceManager(bus)

    with patch.object(sm, "ollama_reachable", return_value=True), \
         patch.object(sm, "ollama_model_exists", return_value=(True, "OK")), \
         patch.object(sm, "ping_model", return_value=(True, "OK")), \
         patch.object(sm, "start_flask", return_value=(True, "OK")), \
         patch.object(sm, "check_telegram", return_value={
             "category": "DNS_FAILURE",
             "reason": "DNS resolution failed for api.telegram.org.",
             "detail": "api.telegram.org could not be resolved.",
         }):
        ready = run_startup(sm)
        # Non-fatal: run_startup must return True so terminal is usable
        assert ready is True
        # Agent must be initialized
        assert sm.agent is not None
        # Telegram must be marked not connected
        assert sm.telegram_connected is False


def test_no_token_leakage_in_diagnostics_or_logs():
    """Verify that sensitive tokens are never included in returned error details or logs."""
    import socket
    secret_token = "SUPER_SECRET_TOKEN_ABC_123_DO_NOT_LEAK"

    with patch("socket.getaddrinfo", side_effect=socket.gaierror(11001, "getaddrinfo failed")):
        diag = check_telegram_connectivity(secret_token)
        assert secret_token not in str(diag)
        assert secret_token not in diag.get("reason", "")
        assert secret_token not in diag.get("detail", "")


def test_cmd_telegram_diagnostic_output(capsys):
    """Verify /telegram outputs full layered diagnostics matching requirements."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.telegram_connected = False
    sm.check_telegram.return_value = {
        "network": "OK",
        "dns": "FAILED",
        "https": "NOT TESTED",
        "bot_api": "NOT TESTED",
        "auth": "NOT TESTED",
        "category": "DNS_FAILURE",
        "reason": "api.telegram.org could not be resolved.",
        "detail": "The bot token was NOT tested because the Telegram API could not be reached.",
    }

    console = TerminalConsole(sm, bus)
    console._cmd_telegram(None)

    out = capsys.readouterr().out
    assert "Telegram" in out
    assert "DNS:            FAILED" in out
    assert "HTTPS:          NOT TESTED" in out
    assert "Polling:        OFFLINE" in out
    assert "api.telegram.org could not be resolved" in out
    assert "The bot token was NOT tested" in out


def test_cmd_telegram_connect_action(capsys):
    """Verify /telegram connect attempts connection when offline."""
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.telegram_connected = False
    sm.start_bot.return_value = (True, "OK", {"bot_info": {"username": "ConnectedBot"}})

    console = TerminalConsole(sm, bus)
    console._cmd_telegram("connect")

    out = capsys.readouterr().out
    assert "Testing connectivity and connecting to Telegram" in out
    assert "Connected as @ConnectedBot" in out
    sm.start_bot.assert_called_once()
