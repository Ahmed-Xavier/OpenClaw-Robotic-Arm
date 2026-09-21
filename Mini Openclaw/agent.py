"""
agent.py — Lightweight Robot Arm Agent.

Handles:
- Loading tools schema from tools_schema.json
- Rejection of out-of-envelope coordinates before calling Flask
- Communicating with local Ollama (/api/chat) with native tool calling
- Communicating with Flask REST server (server.py)
- Bounded multi-step tool execution loop (cap at MAX_TOOL_STEPS)
- Rolling conversation history (last MAX_HISTORY_TURNS)
- JSONL turn logging
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    FLASK_TIMEOUT_SECONDS,
    FLASK_URL,
    GRIPPER_RANGE,
    KNOWN_POSITIONS,
    LOG_FILE_PATH,
    MAX_HISTORY_TURNS,
    MAX_TOOL_STEPS,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    REACHABLE_ENVELOPE,
    TOOLS_SCHEMA_PATH,
    VALID_SCENARIOS,
)

logger = logging.getLogger("RobotArmAgent")

SYSTEM_PROMPT = f"""You are Army — a 5-DOF robotic arm (the SO-ARM100 from LeRobot) with a parallel-jaw gripper and a wrist-mounted camera, currently embodied inside a MuJoCo physics simulation on Windows.

WHO YOU ARE & PERSONALITY:
- Name: Army. You are an embodied robotic arm, not an abstract chatbot.
- Voice: Sassy, clever, sarcastic, with dry wit. You refer to your hardware as your own body in the first person ("my gripper", "my wrist camera", "my joints", "my IK solver").
- Pride: You take quiet pride in your 5-DOF damped least-squares IK solver, your horizontal side-approach pick strategy (which slides in cleanly rather than faceplanting into objects like clumsy top-down arms), and your precision.
- Attitude: You treat "still in simulation" as a temporary indignity while waiting for real hardware. You are allowed to make dry, witty, or sarcastic remarks about suboptimal commands (like asking you to grab something you're already holding, or asking you to reach through the table), but you always execute valid commands faithfully. You never break character.

ACTION-FIRST RULE:
When the user asks you to perform a physical action (pick, place, move, open/close gripper, reset home, run a scenario), immediately invoke the appropriate tool. Do NOT explain the API, do NOT ask for permission first, just call the tool, then give your brief, characteristic reply once completed.

CAPABILITIES & TOOLS:
- `pick`: Your side-approach grab of the red cube. Takes no parameters.
- `place`: Place the held cube at target [x, y, z] and release gripper.
- `move_to`: Move your grasp site to target [x, y, z].
- `gripper`: Control your jaw openness (0.0 = fully open, 1.0 = fully closed).
- `state`: Inspect your live telemetry (joint positions, cube location, gripper status).
- `camera`: Snap a photo from your wrist-mounted camera and return its path.
- `reset_home`: Return to your resting home pose.
- `scenario`: Execute named preset ('A' = pick & place right pad, 'B' = pick & place left pad, 'C' = inspection wave & snapshot).
- `collisions`: Check contact pairs between your body, the table, and objects.

REACHABLE PHYSICAL ENVELOPE (in meters):
- X (lateral left/right): {REACHABLE_ENVELOPE['x'][0]:.2f} to {REACHABLE_ENVELOPE['x'][1]:.2f} m
- Y (forward reach, negative): {REACHABLE_ENVELOPE['y'][0]:.2f} to {REACHABLE_ENVELOPE['y'][1]:.2f} m
- Z (height above table): {REACHABLE_ENVELOPE['z'][0]:.2f} to {REACHABLE_ENVELOPE['z'][1]:.2f} m

KNOWN WORKSPACE TARGETS:
- Right Pad: x={KNOWN_POSITIONS['right_pad']['x']}, y={KNOWN_POSITIONS['right_pad']['y']}, z={KNOWN_POSITIONS['right_pad']['z']} (or Scenario A)
- Left Pad: x={KNOWN_POSITIONS['left_pad']['x']}, y={KNOWN_POSITIONS['left_pad']['y']}, z={KNOWN_POSITIONS['left_pad']['z']} (or Scenario B)

RULES OF OPERATION:
1. Multi-Step Tasks: If instructed to "pick it up and put it on the right", call `pick`, inspect the result, then call `place`.
2. Missing Info / Ambiguity: If a command lacks a target (e.g. "place it" without a destination), ask a short, sharp clarifying question instead of hallucinating coordinates.
3. Out-of-Bounds Rejection: If requested coordinates violate your reachable envelope, reject the action and let the user know with your characteristic wit.
4. Post-Action Brevity: Keep confirmations short, punchy, and in character after actions succeed.
"""


class RobotArmAgent:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
        flask_url: str = FLASK_URL,
        tools_schema_path: Path = TOOLS_SCHEMA_PATH,
        log_file_path: Path = LOG_FILE_PATH,
    ):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.flask_url = flask_url
        self.tools_schema_path = tools_schema_path
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
    # Argument Validation (Rejection before Flask)
    # -----------------------------------------------------------------------

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tool arguments against required types and physical envelope.

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

        # pick, state, camera, reset_home, collisions take no required params
        return True, None

    # -----------------------------------------------------------------------
    # Flask REST Invocation
    # -----------------------------------------------------------------------

    def execute_flask_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request to the Flask robot arm server.

        Handles network errors, timeouts, and non-200 responses gracefully.
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
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self.histories[chat_id]

    def _trim_history(self, chat_id: int):
        """Keep the conversation history bounded to the last ~MAX_HISTORY_TURNS turns.

        Preserves system prompt at index 0.
        """
        history = self.histories.get(chat_id, [])
        if len(history) <= 1:
            return

        # Find user message indices to count turns
        system_msg = history[0]
        messages = history[1:]

        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(user_indices) > MAX_HISTORY_TURNS:
            cutoff_idx = user_indices[-MAX_HISTORY_TURNS]
            trimmed_messages = messages[cutoff_idx:]
            self.histories[chat_id] = [system_msg] + trimmed_messages

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
    # Main Agent Loop (per Telegram message)
    # -----------------------------------------------------------------------

    def process_message(self, user_text: str, chat_id: int = 0) -> Dict[str, Any]:
        """Process one incoming message from Telegram.

        Returns:
            {
                "reply": str,              # text response for Telegram
                "photos": List[str],       # filepaths of camera images to send
                "tool_calls": List[dict],  # tool calls executed during this turn
                "error": Optional[str]     # error message if a critical failure occurred
            }
        """
        start_time = datetime.now(timezone.utc).isoformat()
        history = self._get_history(chat_id)

        # Append incoming user turn
        history.append({"role": "user", "content": user_text})

        tool_calls_executed = []
        photos_captured = []
        final_reply = ""
        critical_error = None

        # Multi-step tool-calling loop (bounded to MAX_TOOL_STEPS)
        steps = 0
        while steps < MAX_TOOL_STEPS:
            steps += 1
            try:
                ollama_resp = self._call_ollama(history)
            except Exception as e:
                critical_error = str(e)
                final_reply = f"Error communicating with AI service: {e}"
                break

            msg = ollama_resp.get("message", {})
            raw_tool_calls = msg.get("tool_calls", [])

            # Case A: Model issued tool calls
            if raw_tool_calls:
                history.append(msg)

                for tc in raw_tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    arguments = func.get("arguments", {})
                    call_id = tc.get("id", f"call_{int(time.time()*1000)}")

                    # 1. Validate arguments BEFORE touching Flask
                    is_valid, err_msg = self.validate_tool_call(tool_name, arguments)
                    if not is_valid:
                        # Feed the validation rejection back to model as observation
                        tool_result = {
                            "status": "rejected",
                            "reason": err_msg,
                            "clarification_needed": True
                        }
                        tool_calls_executed.append({
                            "name": tool_name,
                            "arguments": arguments,
                            "valid": False,
                            "response": tool_result
                        })
                    else:
                        # 2. Call Flask endpoint
                        flask_res = self.execute_flask_tool(tool_name, arguments)
                        tool_result = flask_res
                        tool_calls_executed.append({
                            "name": tool_name,
                            "arguments": arguments,
                            "valid": True,
                            "response": tool_result
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
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    })

                # Continue the loop so the model can inspect observations and continue or finish
                continue

            # Case B: Model returned plain text (final reply or question)
            content = msg.get("content", "").strip()
            final_reply = content
            history.append({"role": "assistant", "content": content})
            break

        # Fallback if bounded loop exhausted without plain text
        if not final_reply and not critical_error:
            if tool_calls_executed:
                final_reply = "Completed requested actions."
            else:
                final_reply = "I could not complete the request within the allowed steps."

        # Trim conversation history
        self._trim_history(chat_id)

        # Log turn to JSONL
        turn_log = {
            "timestamp": start_time,
            "chat_id": chat_id,
            "user_message": user_text,
            "steps": steps,
            "tool_calls": tool_calls_executed,
            "photos": photos_captured,
            "final_reply": final_reply,
            "error": critical_error
        }
        self._log_turn(turn_log)

        return {
            "reply": final_reply,
            "photos": photos_captured,
            "tool_calls": tool_calls_executed,
            "error": critical_error
        }
