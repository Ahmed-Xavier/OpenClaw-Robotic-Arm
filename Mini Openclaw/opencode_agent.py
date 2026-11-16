"""
opencode_agent.py - RobotArmAgent-compatible wrapper around the OpenCode CLI.

Each chat_id keeps its own OpenCode session id, so conversation memory survives
/model switches, /restart, and process restarts.  Messages are sent with:

    opencode run --agent army --format json [--session <id>] "<text>"

The wrapper deliberately mirrors the result contract used by bot.py and
launcher.py for RobotArmAgent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import (
    OPENCODE_AGENT,
    OPENCODE_BIN,
    OPENCODE_TIMEOUT_SECONDS,
)

logger = logging.getLogger("OpenCodeAgent")

FAILURE_MODEL_FAILED_TO_PLAN = "MODEL_FAILED_TO_PLAN"
FAILURE_MODEL_TIMED_OUT = "MODEL_TIMED_OUT"
FAILURE_TOOL_FAILED = "TOOL_FAILED"

PHYSICAL_ACTION_TOOLS = {
    "move_to", "pick", "place", "gripper", "reset_home", "go_home", "scenario",
}
OBSERVATION_TOOLS = {"state", "camera", "collisions"}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BASE_DIR.parent
_DEFAULT_SESSIONS_PATH = _BASE_DIR / "opencode_sessions.json"


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _shim_exe_candidates(shim_path: Path) -> List[Path]:
    """Return real .exe candidates behind an npm shim, in resolution order."""
    base = shim_path.parent
    candidates = [
        base / "node_modules" / "opencode-ai" / "bin" / "opencode.exe",
        base / "node_modules" / "opencode-ai" / "node_modules" / "opencode-windows-x64" / "bin" / "opencode.exe",
        base / "node_modules" / "opencode-ai" / "node_modules" / "opencode-windows-x64-baseline" / "bin" / "opencode.exe",
    ]

    # The npm shim itself names the target (usually the first candidate above).
    # Parse it as a fallback so non-standard installs still work.
    try:
        content = shim_path.read_text(encoding="utf-8", errors="replace")
        for raw in re.findall(r'"([^"\r\n]+?\.exe)"', content, re.I):
            expanded = raw.replace("%~dp0%", str(base)).replace("%dp0%", str(base))
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = base / candidate
            candidates.append(candidate)
    except Exception:
        pass

    ordered: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def _exe_runs(candidate: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(candidate), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def resolve_opencode_cmd(opencode_bin: Optional[str] = None) -> str:
    """Return the real opencode.exe path, never a .cmd/.bat/.ps1 shim.

    Resolution order:
      1. OPENCODE_BIN when it points at an existing .exe
      2. shutil.which(OPENCODE_BIN) when it already resolves to an .exe
      3. the npm shim's folder + known opencode-ai .exe locations
      4. .exe paths referenced inside the shim itself

    The first candidate whose ``--version`` succeeds wins.  If only a shim
    exists, raise FileNotFoundError: running .cmd would route user text
    through cmd.exe and is unsafe.
    """
    configured = str(opencode_bin if opencode_bin is not None else OPENCODE_BIN).strip() or "opencode"
    explicit = Path(configured)
    explicit_suffix = explicit.suffix.lower()

    if explicit_suffix == ".exe" and explicit.is_file():
        return str(explicit.resolve())

    found = shutil.which(configured)
    if found:
        found_path = Path(found)
        found_suffix = found_path.suffix.lower()
        if found_suffix == ".exe":
            return str(found_path)
        if found_suffix not in (".cmd", ".bat", ".ps1"):
            return str(found_path)

        candidates = _shim_exe_candidates(found_path)
        existing = [candidate for candidate in candidates if candidate.is_file()]
        for candidate in existing or candidates:
            if _exe_runs(candidate):
                return str(candidate)

        raise FileNotFoundError(
            f"Could not resolve the real opencode.exe behind shim '{found_path}'. "
            "Install OpenCode or set OPENCODE_BIN to an existing opencode.exe path. "
            "The .cmd shim is never executed for security."
        )

    if explicit_suffix in (".exe", ".cmd", ".bat", ".ps1") or explicit.name.lower() == "opencode":
        raise FileNotFoundError(
            f"opencode binary '{configured}' was not found on PATH. "
            "Install OpenCode or set OPENCODE_BIN to an existing opencode.exe path."
        )

    # Preserve legacy behaviour for custom bare names that may be supplied by
    # tests/callers; subprocess will raise a clear FileNotFoundError if absent.
    return configured


def _empty_result() -> Dict[str, Any]:
    return {
        "reply": "",
        "photos": [],
        "tool_calls": [],
        "error": None,
        "failure_type": None,
        "agent_rounds": 0,
        "physical_tool_calls": 0,
        "physical_actions": 0,
        "observation_calls": 0,
    }


class OpenCodeAgent:
    """Runs one OpenCode session per Telegram/terminal chat id."""

    def __init__(
        self,
        opencode_bin: Optional[str] = None,
        agent_name: Optional[str] = None,
        timeout: Optional[int] = None,
        sessions_path: Optional[Path] = None,
        cwd: Optional[Path] = None,
    ):
        self.opencode_bin = opencode_bin or OPENCODE_BIN
        self.agent_name = agent_name or OPENCODE_AGENT
        self.timeout = int(timeout if timeout is not None else OPENCODE_TIMEOUT_SECONDS)
        self.sessions_path = Path(sessions_path) if sessions_path else _DEFAULT_SESSIONS_PATH
        self.cwd = str(cwd) if cwd else str(_REPO_ROOT)

        # RobotArmAgent exposes .histories; keep a compatible dummy mapping.
        self.histories: Dict[Any, List[Dict[str, Any]]] = {}

        # chat_id -> opencode session id
        self.sessions: Dict[Any, str] = {}
        self._load_sessions()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_key(key: Any) -> Any:
        if isinstance(key, str):
            stripped = key.strip()
            if stripped.lstrip("-").isdigit():
                return int(stripped)
            return stripped
        return key

    def _load_sessions(self) -> None:
        try:
            if not self.sessions_path.exists():
                return
            data = json.loads(self.sessions_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            for key, value in data.items():
                if isinstance(value, str) and value.strip():
                    self.sessions[self._normalise_key(key)] = value.strip()
        except Exception as exc:
            logger.warning("Could not load OpenCode sessions: %s", exc)

    def _save_sessions(self) -> None:
        try:
            self.sessions_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                str(key): value for key, value in self.sessions.items()
            }
            tmp_path = self.sessions_path.with_name(
                self.sessions_path.name + ".tmp"
            )
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(str(tmp_path), str(self.sessions_path))
        except Exception as exc:
            logger.warning("Could not save OpenCode sessions: %s", exc)

    def _drop_session(self, chat_id: int) -> None:
        for key in list(self.sessions):
            if key == chat_id or str(key) == str(chat_id):
                self.sessions.pop(key, None)
        self._save_sessions()

    def reset_conversation(self, chat_id: int) -> None:
        """Forget the OpenCode session for this chat (/new starts fresh)."""
        self._drop_session(chat_id)
        self.histories.pop(chat_id, None)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def _resolve_executable(self) -> List[str]:
        """Return the real opencode .exe resolved by resolve_opencode_cmd()."""
        return [resolve_opencode_cmd(self.opencode_bin)]

    def _build_command(self, text: str, session_id: Optional[str]) -> List[str]:
        args = [
            "run",
            "--agent", self.agent_name,
            "--format", "json",
        ]
        if session_id:
            args.extend(["--session", session_id])
        args.append(text)
        return self._resolve_executable() + args

    def _run_once(self, text: str, session_id: Optional[str]):
        command = self._build_command(text, session_id)
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
            cwd=self.cwd,
            shell=False,
        )

    @staticmethod
    def _session_rejected(stdout: str, stderr: str) -> bool:
        combined = _strip_ansi(f"{stdout}\n{stderr}").lower()
        return (
            "session not found" in combined
            or "session does not exist" in combined
            or "unknown session" in combined
        )

    # ------------------------------------------------------------------
    # JSON event parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_events(stdout: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @staticmethod
    def _event_session_id(event: Dict[str, Any]) -> Optional[str]:
        for container in (event, event.get("part") or {}):
            if not isinstance(container, dict):
                continue
            for key in ("sessionID", "sessionId", "session_id"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _extract_reply(events: List[Dict[str, Any]]) -> str:
        groups: List[List[Any]] = []
        for event in events:
            if event.get("type") != "text":
                continue
            part = event.get("part") or {}
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            message_id = part.get("messageID") or event.get("messageID")
            if message_id is None:
                if groups and groups[-1][0] is None:
                    groups[-1][1] += text
                else:
                    groups.append([None, text])
            elif groups and groups[-1][0] == message_id:
                groups[-1][1] += text
            else:
                groups.append([message_id, text])
        if not groups:
            return ""
        return str(groups[-1][1]).strip()

    @staticmethod
    def _find_existing_path(value: Any, depth: int = 0) -> Optional[str]:
        if depth > 4:
            return None
        if isinstance(value, dict):
            for key in ("path", "file", "image", "photo", "screenshot", "output"):
                candidate = value.get(key)
                found = OpenCodeAgent._find_existing_path(candidate, depth + 1)
                if found:
                    return found
            for candidate in value.values():
                found = OpenCodeAgent._find_existing_path(candidate, depth + 1)
                if found:
                    return found
        elif isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.startswith("{") or candidate.startswith("["):
                try:
                    return OpenCodeAgent._find_existing_path(
                        json.loads(candidate), depth + 1
                    )
                except Exception:
                    return None
            if (
                (candidate.endswith(".png") or candidate.endswith(".jpg")
                 or candidate.endswith(".jpeg"))
                and os.path.exists(candidate)
            ):
                return candidate
        return None

    def _parse_tools(
        self,
        events: List[Dict[str, Any]],
        status_callback: Optional[Callable[[str, str, Any], None]],
    ):
        tool_calls: List[Dict[str, Any]] = []
        photos: List[str] = []
        started_calls = set()
        seen: set = set()

        def notify(event_name: str, name: str, data: Any) -> None:
            if status_callback is None:
                return
            try:
                status_callback(event_name, name, data)
            except Exception:
                pass

        for event in events:
            event_type = event.get("type")
            part = event.get("part") or {}
            if not isinstance(part, dict):
                continue
            if event_type not in ("tool_use", "tool", "tool_call") and part.get("type") != "tool":
                continue
            if part.get("type") not in (None, "tool"):
                continue

            tool_name = part.get("tool") or part.get("name")
            if not isinstance(tool_name, str) or not tool_name.startswith("arm_"):
                continue
            short_name = tool_name[len("arm_"):]

            state = part.get("state")
            if isinstance(state, str):
                status = state.lower()
                state = {}
            elif isinstance(state, dict):
                status = str(state.get("status", "")).lower()
            else:
                state = {}
                status = ""

            arguments = state.get("input") if isinstance(state, dict) else {}
            if not isinstance(arguments, dict):
                arguments = {}

            call_id = (
                part.get("callID")
                or part.get("id")
                or event.get("id")
                or f"{short_name}:{len(tool_calls)}"
            )
            state_error = state.get("error") if isinstance(state, dict) else None

            if status in ("pending", "running", "started"):
                if call_id not in started_calls:
                    started_calls.add(call_id)
                    notify("started", short_name, arguments)
                continue

            if status not in ("completed", "error", "failed", ""):
                continue

            marker = (call_id, status)
            if marker in seen:
                continue
            seen.add(marker)

            if call_id not in started_calls:
                started_calls.add(call_id)
                notify("started", short_name, arguments)

            failed = status in ("error", "failed") or bool(state_error)
            if failed:
                notify("failed", short_name, state or {"error": state_error})
            else:
                notify("completed", short_name, state)

            response: Any = state
            if isinstance(state, dict):
                if state.get("output") is not None:
                    response = state.get("output")
                elif state_error:
                    response = {"error": state_error}

            tool_calls.append({
                "name": short_name,
                "arguments": arguments,
                "valid": True,
                "tool_success": not failed,
                "failure_type": FAILURE_TOOL_FAILED if failed else None,
                "response": response,
            })

            if short_name == "camera":
                photo = self._find_existing_path(state)
                if photo and photo not in photos:
                    photos.append(photo)

        return tool_calls, photos

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_message(
        self,
        user_text: str,
        chat_id: int = 0,
        status_callback: Optional[Callable[[str, str, Any], None]] = None,
    ) -> Dict[str, Any]:
        result = _empty_result()

        session_id = self.sessions.get(chat_id)
        completed = None

        try:
            completed = self._run_once(user_text, session_id)
        except subprocess.TimeoutExpired:
            result["error"] = (
                f"opencode timed out after {self.timeout}s; process killed."
            )
            result["reply"] = result["error"]
            result["failure_type"] = FAILURE_MODEL_TIMED_OUT
            return result
        except FileNotFoundError as exc:
            result["error"] = str(exc).strip() or (
                f"opencode binary '{self.opencode_bin}' was not found on PATH. "
                "Install OpenCode or set OPENCODE_BIN."
            )
            result["reply"] = result["error"]
            result["failure_type"] = FAILURE_MODEL_FAILED_TO_PLAN
            return result
        except Exception as exc:
            result["error"] = f"Could not start opencode: {exc}"
            result["reply"] = result["error"]
            result["failure_type"] = FAILURE_MODEL_FAILED_TO_PLAN
            return result

        # A stale session id is recoverable: drop it and retry once fresh.
        if session_id and self._session_rejected(completed.stdout, completed.stderr):
            self._drop_session(chat_id)
            session_id = None
            try:
                completed = self._run_once(user_text, None)
            except subprocess.TimeoutExpired:
                result["error"] = (
                    f"opencode timed out after {self.timeout}s; process killed."
                )
                result["reply"] = result["error"]
                result["failure_type"] = FAILURE_MODEL_TIMED_OUT
                return result
            except Exception as exc:
                result["error"] = f"Could not start opencode: {exc}"
                result["reply"] = result["error"]
                result["failure_type"] = FAILURE_MODEL_FAILED_TO_PLAN
                return result

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            detail = _strip_ansi(stderr).strip() or _strip_ansi(stdout).strip()
            detail = detail or f"exit code {completed.returncode}"
            result["error"] = (
                f"opencode run failed (exit {completed.returncode}): {detail}"
            )
            result["reply"] = result["error"]
            result["failure_type"] = FAILURE_MODEL_FAILED_TO_PLAN
            return result

        events = self._parse_events(stdout)
        reply = self._extract_reply(events)
        if not reply:
            reply = _strip_ansi(stdout).strip()
        result["reply"] = reply

        new_session_id = None
        for event in events:
            new_session_id = self._event_session_id(event)
            if new_session_id:
                break
        if new_session_id:
            if self.sessions.get(chat_id) != new_session_id:
                self.sessions[chat_id] = new_session_id
                self._save_sessions()

        tool_calls, photos = self._parse_tools(events, status_callback)
        result["tool_calls"] = tool_calls
        result["photos"] = photos
        result["agent_rounds"] = sum(
            1 for event in events if event.get("type") == "step_start"
        )
        result["physical_tool_calls"] = sum(
            1 for call in tool_calls if call["name"] in PHYSICAL_ACTION_TOOLS
        )
        result["physical_actions"] = result["physical_tool_calls"]
        result["observation_calls"] = sum(
            1 for call in tool_calls if call["name"] in OBSERVATION_TOOLS
        )
        return result
