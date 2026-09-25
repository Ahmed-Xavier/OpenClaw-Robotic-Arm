"""
agent.py — Lightweight Robot Arm Agent.

Handles:
- Loading tools schema from tools_schema.json
- Rejection of out-of-envelope coordinates before calling Flask (intent validation)
- Communicating with local Ollama (/api/chat) with native tool calling
- Communicating with Flask REST server (server.py)
- Bounded multi-step tool execution loop
  - MAX_AGENT_ROUNDS: Ollama reasoning invocations
  - MAX_TOOL_CALLS_PER_TURN: physical tool calls
- Rolling conversation history (last MAX_HISTORY_TURNS)
- JSONL turn logging with explicit failure types
- Optional status_callback for Telegram UX (Phases 9, 10, 16)
- reset_conversation() for /new (Phase 11)
- resolve_semantic_target() for semantic world model (Phase 15)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from config import (
    FLASK_TIMEOUT_SECONDS,
    FLASK_URL,
    GRIPPER_RANGE,
    KNOWN_POSITIONS,
    LOG_FILE_PATH,
    MAX_AGENT_ROUNDS,
    MAX_HISTORY_TURNS,
    MAX_TOOL_CALLS_PER_TURN,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    REACHABLE_ENVELOPE,
    SOUL_FILE_PATH,
    TOOLS_SCHEMA_PATH,
    VALID_SCENARIOS,
)

logger = logging.getLogger("RobotArmAgent")

# ---------------------------------------------------------------------------
# Phase 10 — Explicit failure type constants
# ---------------------------------------------------------------------------
# These are used in the returned result dict and in turns.jsonl.
# They distinguish MODEL failures from TOOL failures from PHYSICAL failures.

FAILURE_MODEL_FAILED_TO_PLAN = "MODEL_FAILED_TO_PLAN"
FAILURE_MODEL_TIMED_OUT = "MODEL_TIMED_OUT"
FAILURE_TOOL_REJECTED = "TOOL_REJECTED"          # agent-side validation rejected the call
FAILURE_TOOL_FAILED = "TOOL_FAILED"              # physical action returned success=false
FAILURE_EXECUTION_INCOMPLETE = "EXECUTION_INCOMPLETE"  # limits reached before finishing
FAILURE_FINAL_RESPONSE_FAILED = "FINAL_RESPONSE_FAILED"  # action OK, LLM reply timed out

# Tool classification: observation probes vs motor actions
OBSERVATION_TOOLS = {"state", "camera", "collisions"}
PHYSICAL_ACTION_TOOLS = {"move_to", "pick", "place", "gripper", "reset_home", "scenario"}


def load_soul(soul_path: Path = SOUL_FILE_PATH) -> str:
    """Load the robot arm personality and system prompt from a separate file."""
    if soul_path.exists():
        with open(soul_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    logger.warning("Soul file not found at %s; using default fallback.", soul_path)
    return "You are Army, a 5-DOF robotic arm. Control the arm precisely and stay in character."


# Module-level default system prompt
SYSTEM_PROMPT = load_soul()


# ---------------------------------------------------------------------------
# Phase 15 — Semantic world model helper
# ---------------------------------------------------------------------------

def resolve_semantic_target(name: str) -> Optional[Dict[str, float]]:
    """Resolve a semantic location name to absolute coordinates.

    Example:
        resolve_semantic_target("right_pad")
        -> {"x": 0.15, "y": -0.18, "z": 0.015}

    Returns None if the name is unknown.
    """
    key = name.strip().lower().replace(" ", "_")
    return KNOWN_POSITIONS.get(key)


class RobotArmAgent:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
        flask_url: str = FLASK_URL,
        tools_schema_path: Path = TOOLS_SCHEMA_PATH,
        soul_path: Path = SOUL_FILE_PATH,
        log_file_path: Path = LOG_FILE_PATH,
    ):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.flask_url = flask_url
        self.tools_schema_path = tools_schema_path
        self.soul_path = soul_path
        self.system_prompt = load_soul(self.soul_path)
        self.log_file_path = log_file_path

        # Load tools schema
        self.tools = self._load_tools_schema()

        # Rolling conversation history: dict of chat_id -> list of message dicts
        self.histories: Dict[int, List[Dict[str, Any]]] = {}

        # Reusable HTTP session for efficiency
        self.http_session = requests.Session()

    def _load_tools_schema(self) -> List[Dict[str, Any]]:
        """Load the JSON schema describing robot arm tools."""
        if not self.tools_schema_path.exists():
            raise FileNotFoundError(f"Tools schema file not found at {self.tools_schema_path}")
        with open(self.tools_schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------------------
    # Argument Validation (intent-level rejection before Flask)
    # -----------------------------------------------------------------------

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tool arguments against required types and physical envelope.

        This is INTENT validation — the agent's first gate.
        RobotAPI performs independent PHYSICAL safety validation regardless.

        Rejects invalid calls with an explanatory message. Never silently clamps.
        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        valid_tools = {
            "move_to", "pick", "place", "gripper",
            "state", "camera", "reset_home", "scenario", "collisions"
        }
        if tool_name not in valid_tools:
            return False, f"Unknown tool '{tool_name}'. Available tools: {sorted(list(valid_tools))}"

        # Coordinate-based tools: move_to, place
        if tool_name in ("move_to", "place"):
            for coord in ("x", "y", "z"):
                if coord not in arguments:
                    return False, f"Missing required parameter '{coord}' for {tool_name}."
                try:
                    val = float(arguments[coord])
                    arguments[coord] = val  # ensure clean float
                except (TypeError, ValueError):
                    return False, f"Parameter '{coord}' must be a valid number, got '{arguments.get(coord)}'."

            x, y, z = arguments["x"], arguments["y"], arguments["z"]
            x_min, x_max = REACHABLE_ENVELOPE["x"]
            y_min, y_max = REACHABLE_ENVELOPE["y"]
            z_min, z_max = REACHABLE_ENVELOPE["z"]

            if not (x_min <= x <= x_max):
                return False, (
                    f"Target X coordinate {x:+.3f}m is outside the reachable envelope "
                    f"[{x_min:.2f}, {x_max:.2f}]m. Action rejected."
                )
            if not (y_min <= y <= y_max):
                return False, (
                    f"Target Y coordinate {y:+.3f}m is outside the reachable envelope "
                    f"[{y_min:.2f}, {y_max:.2f}]m. Action rejected."
                )
            if not (z_min <= z <= z_max):
                return False, (
                    f"Target Z coordinate {z:+.3f}m is outside the reachable envelope "
                    f"[{z_min:.2f}, {z_max:.2f}]m. Action rejected."
                )

        # Gripper tool
        elif tool_name == "gripper":
            if "value" not in arguments:
                return False, "Missing required parameter 'value' for gripper."
            try:
                val = float(arguments["value"])
                arguments["value"] = val
            except (TypeError, ValueError):
                return False, f"Gripper 'value' must be a number between 0.0 and 1.0, got '{arguments.get('value')}'."

            g_min, g_max = GRIPPER_RANGE
            if not (g_min <= val <= g_max):
                return False, f"Gripper value {val} is outside valid range [{g_min}, {g_max}]. Action rejected."

        # Scenario tool
        elif tool_name == "scenario":
            if "name" not in arguments:
                return False, "Missing required parameter 'name' for scenario (must be 'A', 'B', or 'C')."
            name = str(arguments["name"]).strip().upper()
            if name not in VALID_SCENARIOS:
                return False, f"Scenario '{name}' is unknown. Must be one of {sorted(list(VALID_SCENARIOS))}."
            arguments["name"] = name

        # Pick tool: optional target parameter ("cube" or "sphere", or natural synonyms)
        elif tool_name == "pick":
            if "target" in arguments and arguments["target"] is not None:
                raw_target = str(arguments["target"]).strip().lower()
                clean_target = "_".join(raw_target.replace("-", " ").split())
                valid_cube = {"cube", "red_cube", "red_box", "box"}
                valid_sphere = {"sphere", "blue_sphere", "ball", "blue_ball"}
                if clean_target not in valid_cube and clean_target not in valid_sphere:
                    return False, f"Invalid pick target '{arguments.get('target')}'. Must be 'cube' or 'sphere'."
                arguments["target"] = "sphere" if clean_target in valid_sphere else "cube"

        # state, camera, reset_home, collisions take no required params
        return True, None

    # -----------------------------------------------------------------------
    # Flask REST Invocation
    # -----------------------------------------------------------------------

    def execute_flask_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request to the Flask robot arm server.

        Handles network errors, timeouts, and non-200 responses gracefully.
        Returns the JSON dict from Flask, or an error dict on failure.
        """
        endpoint = f"{self.flask_url}/{tool_name}"
        try:
            if tool_name in ("state", "camera", "collisions"):
                resp = self.http_session.get(endpoint, timeout=FLASK_TIMEOUT_SECONDS)
            else:
                resp = self.http_session.post(endpoint, json=arguments, timeout=FLASK_TIMEOUT_SECONDS)

            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    return {"status": "ok", "raw_response": resp.text}
            else:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", resp.text)
                except Exception:
                    err_msg = resp.text
                return {"error": f"Robot arm server error (HTTP {resp.status_code}): {err_msg}"}

        except requests.exceptions.ConnectionError:
            return {"error": f"Failed to connect to robot arm server at {self.flask_url}. Is server.py running?"}
        except requests.exceptions.Timeout:
            return {"error": f"Robot arm server at {self.flask_url} timed out after {FLASK_TIMEOUT_SECONDS}s."}
        except Exception as exc:
            return {"error": f"Unexpected error communicating with robot server: {exc}"}

    # -----------------------------------------------------------------------
    # Ollama Chat Invocation
    # -----------------------------------------------------------------------

    def _call_ollama(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Ollama /api/chat with native tool calling."""
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "tools": self.tools,
            "stream": False,
        }
        url = f"{self.ollama_url}/api/chat"
        try:
            resp = self.http_session.post(
                url,
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Ollama at {self.ollama_url}. Is Ollama running?")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timed out after {OLLAMA_TIMEOUT_SECONDS}s.")

    # -----------------------------------------------------------------------
    # Conversation History Management
    # -----------------------------------------------------------------------

    def _get_history(self, chat_id: int) -> List[Dict[str, Any]]:
        """Get or initialize rolling history for this chat."""
        if chat_id not in self.histories:
            self.histories[chat_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
        return self.histories[chat_id]

    def _trim_history(self, chat_id: int):
        """Keep the conversation history bounded to the last ~MAX_HISTORY_TURNS turns.

        Preserves system prompt at index 0.
        """
        history = self.histories.get(chat_id, [])
        if len(history) <= 1:
            return

        system_msg = history[0]
        messages = history[1:]

        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(user_indices) > MAX_HISTORY_TURNS:
            cutoff_idx = user_indices[-MAX_HISTORY_TURNS]
            trimmed_messages = messages[cutoff_idx:]
            self.histories[chat_id] = [system_msg] + trimmed_messages

    # -----------------------------------------------------------------------
    # Phase 11 — /new conversation reset
    # -----------------------------------------------------------------------

    def reset_conversation(self, chat_id: int) -> None:
        """Clear conversation history for this chat.

        Called directly by bot.py when the user sends /new.
        Must NOT go through Qwen.
        """
        self.histories[chat_id] = [
            {"role": "system", "content": self.system_prompt}
        ]
        logger.info("Conversation history reset for chat %s.", chat_id)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_turn(self, turn_record: Dict[str, Any]):
        """Append a human-readable JSON line to the turn log file."""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn_record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to write turn log: %s", e)

    # -----------------------------------------------------------------------
    # Phase 10 — Human-readable action label for status callbacks
    # -----------------------------------------------------------------------

    @staticmethod
    def _action_label(tool_name: str, arguments: Dict[str, Any]) -> str:
        """Map a tool call to a user-facing status string."""
        labels = {
            "pick": "Picking up the cube...",
            "place": "Placing the cube...",
            "move_to": "Moving to position...",
            "gripper": "Adjusting gripper...",
            "reset_home": "Returning to home position...",
            "state": "Checking state...",
            "camera": "Capturing camera image...",
            "collisions": "Checking collisions...",
            "scenario": f"Running scenario {arguments.get('name', '')}...",
        }
        return labels.get(tool_name, f"Executing {tool_name}...")

    # -----------------------------------------------------------------------
    # Main Agent Loop (per Telegram message / CLI message)
    # -----------------------------------------------------------------------

    def process_message(
        self,
        user_text: str,
        chat_id: int = 0,
        status_callback: Optional[Callable[[str, str, Any], None]] = None,
    ) -> Dict[str, Any]:
        """Process one incoming message.

        Args:
            user_text:       The user's message text.
            chat_id:         Telegram chat ID (or 0 for CLI).
            status_callback: Optional callback(event, tool_name, data) for
                             real-time Telegram status updates (Phase 16).
                             Events: "started", "completed", "failed".

        Returns:
            {
                "reply":               str,         # text for the user
                "photos":              List[str],   # camera image paths
                "tool_calls":          List[dict],  # tool calls executed
                "error":               str | None,  # critical failure message
                "failure_type":        str | None,  # Phase 10 failure type
                "agent_rounds":        int,         # Ollama invocations used
                "physical_tool_calls": int,         # robot actions executed
            }
        """
        start_time = datetime.now(timezone.utc).isoformat()
        history = self._get_history(chat_id)

        # Append incoming user turn
        history.append({"role": "user", "content": user_text})

        tool_calls_executed: List[Dict[str, Any]] = []
        photos_captured: List[str] = []
        final_reply = ""
        critical_error = None
        failure_type = None
        agent_rounds = 0
        physical_tool_calls_count = 0
        observation_calls_count = 0

        # ---------------------------------------------------------------
        # Phase 8 — bounded loop with two distinct limits:
        #   MAX_AGENT_ROUNDS:      Ollama reasoning invocations
        #   MAX_TOOL_CALLS_PER_TURN: physical tool calls
        # ---------------------------------------------------------------
        while agent_rounds < MAX_AGENT_ROUNDS:
            agent_rounds += 1

            # Phase 9 — call Ollama; distinguish timeout from other errors
            try:
                ollama_resp = self._call_ollama(history)
            except RuntimeError as e:
                err_str = str(e)
                if "timed out" in err_str.lower():
                    failure_type = FAILURE_MODEL_TIMED_OUT
                else:
                    failure_type = FAILURE_MODEL_FAILED_TO_PLAN
                critical_error = err_str

                # Phase 9 — if physical work was already done, report it honestly
                if tool_calls_executed:
                    last_tc = tool_calls_executed[-1]
                    last_result = last_tc.get("response", {})
                    if isinstance(last_result, dict) and last_result.get("success"):
                        action = last_result.get("action", last_tc["name"])
                        final_reply = (
                            f"Physical action '{action}' completed successfully. "
                            f"However, I couldn't generate a final response because the model "
                            f"{'timed out' if failure_type == FAILURE_MODEL_TIMED_OUT else 'failed'}. "
                            f"Please check the robot state for details."
                        )
                        failure_type = FAILURE_FINAL_RESPONSE_FAILED
                    else:
                        final_reply = f"Error communicating with AI service: {err_str}"
                else:
                    final_reply = f"Error communicating with AI service: {err_str}"
                break

            msg = ollama_resp.get("message", {})
            raw_tool_calls = msg.get("tool_calls", [])

            # ---------------------------------------------------------------
            # Case A: Model issued tool calls
            # ---------------------------------------------------------------
            if raw_tool_calls:
                history.append(msg)

                for tc in raw_tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    arguments = func.get("arguments", {})
                    call_id = tc.get("id", f"call_{int(time.time()*1000)}")

                    is_physical_action = tool_name in PHYSICAL_ACTION_TOOLS

                    # Enforce physical action limit only on physical motor actions
                    if is_physical_action and physical_tool_calls_count >= MAX_TOOL_CALLS_PER_TURN:
                        logger.warning(
                            "Physical action limit (%d) reached. Stopping physical action execution.",
                            MAX_TOOL_CALLS_PER_TURN,
                        )
                        # Feed limit notice back to model so it knows
                        tool_result = {
                            "success": False,
                            "action": tool_name,
                            "result": None,
                            "error": {
                                "code": "EXECUTION_INCOMPLETE",
                                "message": (
                                    f"Physical action limit ({MAX_TOOL_CALLS_PER_TURN}) "
                                    "reached. No further robot physical actions will be executed."
                                ),
                            },
                        }
                        failure_type = FAILURE_EXECUTION_INCOMPLETE
                        history.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        })
                        tool_calls_executed.append({
                            "name": tool_name,
                            "arguments": arguments,
                            "valid": False,
                            "tool_success": False,
                            "failure_type": FAILURE_EXECUTION_INCOMPLETE,
                            "response": tool_result,
                        })
                        continue

                    # Phase 9/10 — Validate arguments BEFORE touching Flask
                    is_valid, err_msg = self.validate_tool_call(tool_name, arguments)
                    if not is_valid:
                        tool_result = {
                            "success": False,
                            "action": tool_name,
                            "result": None,
                            "error": {
                                "code": "TOOL_REJECTED",
                                "message": err_msg,
                            },
                            "clarification_needed": True,
                        }
                        tool_calls_executed.append({
                            "name": tool_name,
                            "arguments": arguments,
                            "valid": False,
                            "tool_success": False,
                            "failure_type": FAILURE_TOOL_REJECTED,
                            "response": tool_result,
                        })
                    else:
                        # Phase 16 — status callback: action started
                        if status_callback is not None:
                            try:
                                status_callback("started", tool_name, arguments)
                            except Exception:
                                pass

                        # Execute the Flask endpoint
                        flask_res = self.execute_flask_tool(tool_name, arguments)
                        tool_result = flask_res

                        # Determine physical success / failure type
                        tool_success = True
                        tc_failure_type = None

                        if isinstance(flask_res, dict):
                            if "error" in flask_res and "success" not in flask_res:
                                # Flask-level transport error (connection, timeout, etc.)
                                tool_success = False
                                tc_failure_type = FAILURE_TOOL_FAILED
                            elif flask_res.get("success") is False:
                                # Structured RobotAPI failure
                                tool_success = False
                                tc_failure_type = FAILURE_TOOL_FAILED
                                if failure_type is None:
                                    failure_type = FAILURE_TOOL_FAILED

                        # Phase 16 — status callback: completed or failed
                        if status_callback is not None:
                            try:
                                if tool_success:
                                    status_callback("completed", tool_name, flask_res)
                                else:
                                    status_callback("failed", tool_name, flask_res)
                            except Exception:
                                pass

                        if is_physical_action:
                            physical_tool_calls_count += 1
                        else:
                            observation_calls_count += 1

                        tool_calls_executed.append({
                            "name": tool_name,
                            "arguments": arguments,
                            "valid": True,
                            "tool_success": tool_success,
                            "failure_type": tc_failure_type,
                            "response": tool_result,
                        })

                        # If camera snapshot was taken, collect image path
                        if tool_name == "camera" and isinstance(flask_res, dict) and "path" in flask_res:
                            cam_path = flask_res["path"]
                            if os.path.exists(cam_path):
                                photos_captured.append(cam_path)

                    # Feed tool observation back to model
                    history.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })

                # Continue the loop so the model can observe and continue / finish
                continue

            # ---------------------------------------------------------------
            # Case B: Model returned plain text (final reply or question)
            # ---------------------------------------------------------------
            content = msg.get("content", "").strip()
            final_reply = content
            history.append({"role": "assistant", "content": content})
            break

        # ---------------------------------------------------------------
        # Phase 8/9 — Handle loop exhaustion
        # ---------------------------------------------------------------
        if not final_reply and not critical_error:
            if tool_calls_executed:
                # Summarize what actually happened rather than claiming completion
                executed_names = [tc["name"] for tc in tool_calls_executed if tc["valid"]]
                successful = [tc["name"] for tc in tool_calls_executed if tc.get("tool_success")]
                failed = [tc["name"] for tc in tool_calls_executed if not tc.get("tool_success")]

                parts = []
                if successful:
                    parts.append(f"Completed: {', '.join(successful)}")
                if failed:
                    parts.append(f"Failed: {', '.join(failed)}")

                final_reply = (
                    f"Agent round limit reached. {' | '.join(parts)}. "
                    "Request may be incomplete."
                )
                failure_type = failure_type or FAILURE_EXECUTION_INCOMPLETE
            else:
                final_reply = "I could not complete the request within the allowed steps."
                failure_type = failure_type or FAILURE_EXECUTION_INCOMPLETE

        # Trim conversation history
        self._trim_history(chat_id)

        # Phase 19 — Log turn with explicit failure fields
        turn_log = {
            "timestamp": start_time,
            "chat_id": chat_id,
            "user_message": user_text,
            "agent_rounds": agent_rounds,
            "physical_tool_calls_count": physical_tool_calls_count,
            "observation_calls_count": observation_calls_count,
            "tool_calls": [
                {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "valid": tc["valid"],
                    "tool_success": tc.get("tool_success"),
                    "failure_type": tc.get("failure_type"),
                    # Log structured result (not full raw state dumps)
                    "response_success": (
                        tc["response"].get("success")
                        if isinstance(tc["response"], dict)
                        else None
                    ),
                    "response_action": (
                        tc["response"].get("action")
                        if isinstance(tc["response"], dict)
                        else None
                    ),
                    "response_error": (
                        tc["response"].get("error")
                        if isinstance(tc["response"], dict)
                        else None
                    ),
                }
                for tc in tool_calls_executed
            ],
            "photos": photos_captured,
            "final_reply": final_reply,
            "failure_type": failure_type,
            "error": critical_error,
        }
        self._log_turn(turn_log)

        return {
            "reply": final_reply,
            "photos": photos_captured,
            "tool_calls": tool_calls_executed,
            "error": critical_error,
            "failure_type": failure_type,
            "agent_rounds": agent_rounds,
            "physical_tool_calls": physical_tool_calls_count,
            "physical_actions": physical_tool_calls_count,
            "observation_calls": observation_calls_count,
        }
