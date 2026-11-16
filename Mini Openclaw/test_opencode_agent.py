"""Mocked tests for the OpenCode CLI agent and its launcher/MCP integration."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from opencode_agent import (  # noqa: E402
    FAILURE_MODEL_FAILED_TO_PLAN,
    FAILURE_MODEL_TIMED_OUT,
    OpenCodeAgent,
    resolve_opencode_cmd,
)


def _make_npm_layout(tmp_path, baseline=False):
    npm = tmp_path / "npm"
    shim = npm / "opencode.cmd"
    shim.parent.mkdir(parents=True)
    shim.write_text(
        '@ECHO off\r\n"%dp0%\\node_modules\\opencode-ai\\bin\\opencode.exe" %*\r\n',
        encoding="utf-8",
    )
    package = npm / "node_modules" / "opencode-ai"
    if baseline:
        exe = package / "node_modules" / "opencode-windows-x64-baseline" / "bin" / "opencode.exe"
    else:
        exe = package / "bin" / "opencode.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake exe")
    return shim, exe


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["opencode"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _event_stdout(session_id="ses_test_1", text="OK", tools=None):
    events = [
        {
            "type": "step_start",
            "sessionID": session_id,
            "part": {"type": "step-start", "sessionID": session_id},
        }
    ]
    for tool in tools or []:
        events.append(tool)
    events.append({
        "type": "text",
        "sessionID": session_id,
        "part": {
            "type": "text",
            "text": text,
            "messageID": "msg_1",
            "sessionID": session_id,
        },
    })
    events.append({
        "type": "step_finish",
        "sessionID": session_id,
        "part": {"type": "step-finish", "sessionID": session_id},
    })
    return "\n".join(json.dumps(event) for event in events)


def _command_from(call):
    return call.args[0]


def _agent(tmp_path, **kwargs):
    return OpenCodeAgent(
        opencode_bin="fake-opencode",
        sessions_path=tmp_path / "opencode_sessions.json",
        **kwargs,
    )


def test_first_call_has_no_session_and_stores_id(tmp_path):
    agent = _agent(tmp_path)
    run = MagicMock(return_value=_completed(_event_stdout("ses_first")))

    with patch("opencode_agent.subprocess.run", run):
        result = agent.process_message("hello", chat_id=1)

    assert result["reply"] == "OK"
    assert "--session" not in _command_from(run.call_args_list[0])
    assert agent.sessions[1] == "ses_first"
    assert json.loads(agent.sessions_path.read_text(encoding="utf-8"))["1"] == "ses_first"


def test_second_call_for_same_chat_includes_session(tmp_path):
    agent = _agent(tmp_path)
    agent.sessions[1] = "ses_existing"
    run = MagicMock(return_value=_completed(_event_stdout("ses_existing")))

    with patch("opencode_agent.subprocess.run", run):
        agent.process_message("again", chat_id=1)

    command = _command_from(run.call_args_list[0])
    session_index = command.index("--session")
    assert command[session_index + 1] == "ses_existing"


def test_different_chat_gets_its_own_session(tmp_path):
    agent = _agent(tmp_path)
    agent.sessions[1] = "ses_one"
    agent.sessions[2] = "ses_two"
    run = MagicMock(side_effect=[
        _completed(_event_stdout("ses_one")),
        _completed(_event_stdout("ses_two")),
    ])

    with patch("opencode_agent.subprocess.run", run):
        agent.process_message("one", chat_id=1)
        agent.process_message("two", chat_id=2)

    first = _command_from(run.call_args_list[0])
    second = _command_from(run.call_args_list[1])
    assert first[first.index("--session") + 1] == "ses_one"
    assert second[second.index("--session") + 1] == "ses_two"


def test_reset_conversation_makes_next_call_fresh(tmp_path):
    agent = _agent(tmp_path)
    agent.sessions[7] = "ses_old"
    agent.reset_conversation(7)

    run = MagicMock(return_value=_completed(_event_stdout("ses_new")))
    with patch("opencode_agent.subprocess.run", run):
        agent.process_message("fresh", chat_id=7)

    assert "--session" not in _command_from(run.call_args_list[0])
    assert agent.sessions[7] == "ses_new"


def test_rejected_session_retries_once_without_session(tmp_path):
    agent = _agent(tmp_path)
    agent.sessions[5] = "ses_missing"
    run = MagicMock(side_effect=[
        _completed(stderr="Error: Session not found", returncode=1),
        _completed(_event_stdout("ses_replacement")),
    ])

    with patch("opencode_agent.subprocess.run", run):
        result = agent.process_message("hello", chat_id=5)

    assert result["reply"] == "OK"
    assert run.call_count == 2
    first = _command_from(run.call_args_list[0])
    second = _command_from(run.call_args_list[1])
    assert first[first.index("--session") + 1] == "ses_missing"
    assert "--session" not in second
    assert agent.sessions[5] == "ses_replacement"


def test_sessions_file_is_reloaded(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"9": "ses_persisted"}), encoding="utf-8")

    agent = OpenCodeAgent(opencode_bin="fake-opencode", sessions_path=path)
    assert agent.sessions[9] == "ses_persisted"


def test_timeout_returns_model_timed_out(tmp_path):
    agent = _agent(tmp_path, timeout=1)
    run = MagicMock(side_effect=subprocess.TimeoutExpired("opencode", 1))

    with patch("opencode_agent.subprocess.run", run):
        result = agent.process_message("slow", chat_id=1)

    assert result["failure_type"] == FAILURE_MODEL_TIMED_OUT
    assert "timed out" in result["error"].lower()


def test_missing_binary_returns_clear_error(tmp_path):
    agent = OpenCodeAgent(
        opencode_bin="definitely-not-a-real-opencode-binary",
        sessions_path=tmp_path / "sessions.json",
    )
    run = MagicMock(side_effect=FileNotFoundError())

    with patch("opencode_agent.subprocess.run", run):
        result = agent.process_message("hello", chat_id=1)

    assert result["failure_type"] == FAILURE_MODEL_FAILED_TO_PLAN
    assert "not found" in result["error"].lower()


def test_status_callback_receives_started_and_completed(tmp_path):
    agent = _agent(tmp_path)
    tool_event = {
        "type": "tool_use",
        "sessionID": "ses_tool",
        "part": {
            "type": "tool",
            "tool": "arm_state",
            "callID": "call_1",
            "state": {
                "status": "completed",
                "input": {},
                "output": {"success": True, "holding": False},
            },
        },
    }
    run = MagicMock(return_value=_completed(_event_stdout("ses_tool", tools=[tool_event])))
    events = []

    def callback(event, name, data):
        events.append((event, name))

    with patch("opencode_agent.subprocess.run", run):
        result = agent.process_message("state?", chat_id=1, status_callback=callback)

    assert events == [("started", "state"), ("completed", "state")]
    assert result["tool_calls"][0]["name"] == "state"
    assert result["observation_calls"] == 1


def test_launcher_model_accepts_opencode_when_binary_exists():
    from launcher import OPENCODE_MODEL_NAME, EventBus, ServiceManager, TerminalConsole

    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.current_model = "qwen3.5:4b"
    console = TerminalConsole(sm, bus)

    with patch("launcher.shutil.which", return_value="C:/fake/opencode.exe"), \
         patch("launcher.subprocess.run", return_value=_completed("1.19.5")):
        console._cmd_model(OPENCODE_MODEL_NAME)

    sm.set_model.assert_called_once_with(OPENCODE_MODEL_NAME)


def test_launcher_model_rejects_opencode_when_binary_missing(capsys):
    from launcher import OPENCODE_MODEL_NAME, EventBus, ServiceManager, TerminalConsole

    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.current_model = "qwen3.5:4b"
    console = TerminalConsole(sm, bus)

    with patch("launcher.shutil.which", return_value=None):
        console._cmd_model(OPENCODE_MODEL_NAME)

    out = capsys.readouterr().out
    assert "not found on PATH" in out
    sm.set_model.assert_not_called()


def test_mcp_envelope_rejects_out_of_range_values():
    import mcp_arm_server as arm

    with patch.object(arm, "_post") as post:
        for result in (
            arm.move_to(0.50, -0.20, 0.10),
            arm.move_to(0.0, 0.20, 0.10),
            arm.move_to(0.0, -0.20, -0.10),
            arm.place(0.0, -0.20, 0.50),
            arm.gripper(1.5),
            arm.gripper(-0.1),
            arm.scenario("D"),
            arm.pick("cylinder"),
        ):
            assert result["success"] is False
            assert result["error"]["code"] == "TOOL_REJECTED"

    post.assert_not_called()


def test_mcp_timeout_is_at_least_60_seconds():
    import mcp_arm_server as arm

    assert arm.TIMEOUT >= 60


def test_startup_skips_ollama_for_opencode_model():
    from launcher import OPENCODE_MODEL_NAME, EventBus, ServiceManager, run_startup

    bus = EventBus()
    sm = ServiceManager(bus)
    sm._current_model = OPENCODE_MODEL_NAME

    with patch.object(sm, "ollama_reachable") as reachable, \
         patch.object(sm, "ollama_model_exists") as exists, \
         patch.object(sm, "ping_model") as ping, \
         patch.object(sm, "start_flask", return_value=(True, "OK")), \
         patch.object(sm, "start_bot", return_value=(
             True, "disabled", {"category": "NO_TOKEN", "bot_info": {}}
         )), \
         patch("launcher.shutil.which", return_value="C:/fake/opencode.exe"):
        assert run_startup(sm) is True

    reachable.assert_not_called()
    exists.assert_not_called()
    ping.assert_not_called()
    assert isinstance(sm.agent, OpenCodeAgent)


def test_resolver_unwraps_cmd_shim_to_real_exe(tmp_path):
    shim, exe = _make_npm_layout(tmp_path)
    version = MagicMock(return_value=_completed("1.19.5"))

    with patch("opencode_agent.shutil.which", return_value=str(shim)), \
         patch("opencode_agent.subprocess.run", version):
        resolved = resolve_opencode_cmd()

    assert resolved == str(exe)
    assert not resolved.lower().endswith(".cmd")
    assert _command_from(version.call_args_list[0]) == [str(exe), "--version"]


def test_resolver_tries_known_paths_in_order(tmp_path):
    shim, first_exe = _make_npm_layout(tmp_path)
    second_exe = (
        shim.parent / "node_modules" / "opencode-ai" / "node_modules"
        / "opencode-windows-x64" / "bin" / "opencode.exe"
    )
    second_exe.parent.mkdir(parents=True)
    second_exe.write_bytes(b"fake exe")

    def fake_run(command, **kwargs):
        if command[0] == str(first_exe):
            return _completed("", "not supported", returncode=1)
        return _completed("1.19.5")

    with patch("opencode_agent.shutil.which", return_value=str(shim)), \
         patch("opencode_agent.subprocess.run", side_effect=fake_run):
        assert resolve_opencode_cmd() == str(second_exe)


def test_resolver_never_falls_back_to_cmd(tmp_path):
    shim = tmp_path / "npm" / "opencode.cmd"
    shim.parent.mkdir(parents=True)
    shim.write_text("@ECHO off\r\nopencode %*\r\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _completed("", "not found", returncode=1)

    with patch("opencode_agent.shutil.which", return_value=str(shim)), \
         patch("opencode_agent.subprocess.run", side_effect=fake_run):
        with pytest.raises(FileNotFoundError):
            resolve_opencode_cmd()

    assert all(not command[0].lower().endswith((".cmd", ".bat", ".ps1")) for command in calls)


def test_launcher_validation_uses_resolved_exe_not_bare_name(tmp_path):
    from launcher import OPENCODE_MODEL_NAME, EventBus, ServiceManager, TerminalConsole

    shim, exe = _make_npm_layout(tmp_path)
    bus = EventBus()
    sm = MagicMock(spec=ServiceManager)
    sm.current_model = "qwen3.5:4b"
    console = TerminalConsole(sm, bus)

    with patch("launcher.shutil.which", return_value=str(shim)), \
         patch("launcher.subprocess.run", return_value=_completed("1.19.5")) as run:
        console._cmd_model(OPENCODE_MODEL_NAME)

    commands = [_command_from(call) for call in run.call_args_list]
    assert [str(exe), "--version"] in commands
    assert all(command[0] != "opencode" for command in commands)
    assert all(not command[0].lower().endswith(".cmd") for command in commands)
    sm.set_model.assert_called_once_with(OPENCODE_MODEL_NAME)


def test_user_text_is_one_argument_to_real_exe(tmp_path):
    shim, exe = _make_npm_layout(tmp_path)
    payload = "hi & echo pwned | dir"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "--version" in command:
            return _completed("1.19.5")
        return _completed(_event_stdout())

    agent = OpenCodeAgent(
        opencode_bin=str(shim),
        sessions_path=tmp_path / "opencode_sessions.json",
    )
    with patch("opencode_agent.shutil.which", return_value=str(shim)), \
         patch("opencode_agent.subprocess.run", side_effect=fake_run):
        result = agent.process_message(payload, chat_id=1)

    assert result["reply"] == "OK"
    message_command, message_kwargs = calls[-1]
    assert message_command[0] == str(exe)
    assert message_command[-1] == payload
    assert payload not in message_command[:-1]
    assert all(not command[0].lower().endswith((".cmd", ".bat", ".ps1")) for command, _ in calls)
    assert message_kwargs.get("shell") is False

