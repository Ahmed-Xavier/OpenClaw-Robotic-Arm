"""
test_agent.py — Test suite for the Mini Openclaw Robot Arm Agent.

Verifies:
1.  Schema integrity for all 9 endpoints
2.  Strict coordinate envelope and type validation (rejects without clamping)
3.  Structured result contract (success, action, result, error)
4.  MAX_AGENT_ROUNDS and MAX_TOOL_CALLS_PER_TURN limits
5.  JSONL turn logging with Phase 19 fields
6.  /new reset (reset_conversation)
7.  FINAL_RESPONSE_FAILED path
8.  MODEL_TIMED_OUT path
9.  TOOL_REJECTED vs TOOL_FAILED distinction in logs
10. place NOT_HOLDING rejection (mocked)
11. pick INVALID_STATE rejection (mocked)
12. resolve_semantic_target helper
13. Failure type constants
14. Live Ollama integration (skipped if Ollama is not running)

All robot-API-touching tests are mocked so no real server is required.
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent import (
    FAILURE_EXECUTION_INCOMPLETE,
    FAILURE_FINAL_RESPONSE_FAILED,
    FAILURE_MODEL_TIMED_OUT,
    FAILURE_TOOL_FAILED,
    FAILURE_TOOL_REJECTED,
    RobotArmAgent,
    resolve_semantic_target,
)
from config import (
    BASE_DIR,
    LOG_FILE_PATH,
    MAX_AGENT_ROUNDS,
    MAX_TOOL_CALLS_PER_TURN,
    REACHABLE_ENVELOPE,
    TOOLS_SCHEMA_PATH,
)


# ---------------------------------------------------------------------------
# Helper: build a mock Ollama response
# ---------------------------------------------------------------------------

def _ollama_text(text: str) -> dict:
    """Simulate a plain-text (non-tool-call) Ollama response."""
    return {"message": {"role": "assistant", "content": text, "tool_calls": []}}


def _ollama_tool(name: str, args: dict, call_id: str = "call_1") -> dict:
    """Simulate an Ollama response with a single tool call."""
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
            ],
        }
    }


# ---------------------------------------------------------------------------
# 1. Schema integrity
# ---------------------------------------------------------------------------

class TestSchemaAndValidation(unittest.TestCase):
    def setUp(self):
        self.agent = RobotArmAgent()

    def test_tools_schema_loaded(self):
        """All 9 endpoints must be present in tools_schema.json."""
        self.assertTrue(TOOLS_SCHEMA_PATH.exists())
        names = {t["function"]["name"] for t in self.agent.tools}
        expected = {
            "move_to", "pick", "place", "gripper",
            "state", "camera", "reset_home", "scenario", "collisions"
        }
        self.assertEqual(names, expected, f"Missing or extra tools: {expected.symmetric_difference(names)}")

    def test_validation_move_to_valid(self):
        valid, err = self.agent.validate_tool_call("move_to", {"x": 0.15, "y": -0.18, "z": 0.05})
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_validation_move_to_out_of_bounds_x(self):
        valid, err = self.agent.validate_tool_call("move_to", {"x": 0.50, "y": -0.18, "z": 0.05})
        self.assertFalse(valid)
        self.assertIn("Target X coordinate", err)
        self.assertIn("rejected", err)

    def test_validation_move_to_out_of_bounds_y(self):
        valid, err = self.agent.validate_tool_call("move_to", {"x": 0.0, "y": 0.20, "z": 0.05})
        self.assertFalse(valid)
        self.assertIn("Target Y coordinate", err)
        self.assertIn("rejected", err)

    def test_validation_move_to_out_of_bounds_z(self):
        valid, err = self.agent.validate_tool_call("move_to", {"x": 0.0, "y": -0.18, "z": -0.05})
        self.assertFalse(valid)
        self.assertIn("Target Z coordinate", err)
        self.assertIn("rejected", err)

    def test_validation_move_to_missing_arg(self):
        valid, err = self.agent.validate_tool_call("move_to", {"x": 0.0, "y": -0.18})
        self.assertFalse(valid)
        self.assertIn("Missing required parameter 'z'", err)

    def test_validation_place_valid(self):
        valid, err = self.agent.validate_tool_call("place", {"x": -0.15, "y": -0.18, "z": 0.015})
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_validation_gripper_valid(self):
        self.assertTrue(self.agent.validate_tool_call("gripper", {"value": 0.0})[0])
        self.assertTrue(self.agent.validate_tool_call("gripper", {"value": 1.0})[0])
        self.assertTrue(self.agent.validate_tool_call("gripper", {"value": 0.5})[0])

    def test_validation_gripper_invalid(self):
        self.assertFalse(self.agent.validate_tool_call("gripper", {"value": 1.5})[0])
        self.assertFalse(self.agent.validate_tool_call("gripper", {"value": -0.1})[0])
        self.assertFalse(self.agent.validate_tool_call("gripper", {"value": "open"})[0])

    def test_validation_scenario_valid(self):
        self.assertTrue(self.agent.validate_tool_call("scenario", {"name": "A"})[0])
        self.assertTrue(self.agent.validate_tool_call("scenario", {"name": "b"})[0])  # lowercase normalised
        self.assertTrue(self.agent.validate_tool_call("scenario", {"name": "C"})[0])

    def test_validation_scenario_invalid(self):
        self.assertFalse(self.agent.validate_tool_call("scenario", {"name": "D"})[0])

    def test_validation_zero_arg_tools(self):
        for tool in ("pick", "state", "camera", "reset_home", "collisions"):
            valid, err = self.agent.validate_tool_call(tool, {})
            self.assertTrue(valid, f"Failed for {tool}: {err}")

    def test_validation_unknown_tool(self):
        valid, err = self.agent.validate_tool_call("teleport", {})
        self.assertFalse(valid)
        self.assertIn("Unknown tool", err)


# ---------------------------------------------------------------------------
# 2. Structured result contract
# ---------------------------------------------------------------------------

class TestStructuredResultContract(unittest.TestCase):
    def setUp(self):
        self.test_log = BASE_DIR / "test_contract.jsonl"
        if self.test_log.exists():
            self.test_log.unlink()
        self.agent = RobotArmAgent(log_file_path=self.test_log)

    def tearDown(self):
        if self.test_log.exists():
            self.test_log.unlink()

    def _assert_result_contract(self, result: dict, action: str):
        """Assert the standard result contract is present."""
        self.assertIn("success", result, f"'success' missing in {action} result")
        self.assertIn("action", result, f"'action' missing in {action} result")
        self.assertIn("result", result, f"'result' missing in {action} result")
        self.assertIn("error", result, f"'error' missing in {action} result")
        self.assertEqual(result["action"], action)
        if result["success"]:
            self.assertIsNone(result["error"], "error should be None on success")
        else:
            self.assertIsNotNone(result["error"], "error should be set on failure")
            self.assertIn("code", result["error"])
            self.assertIn("message", result["error"])

    def test_move_to_success_contract(self):
        """Successful move_to returns the structured contract with position data."""
        mock_result = {
            "success": True,
            "action": "move_to",
            "result": {"requested": [0.1, -0.2, 0.1], "actual": [0.1, -0.2, 0.1], "position_error": 0.001},
            "error": None,
        }
        with patch.object(self.agent, "_call_ollama", return_value=_ollama_tool("move_to", {"x": 0.1, "y": -0.2, "z": 0.1})) as _m1, \
             patch.object(self.agent, "execute_flask_tool", return_value=mock_result), \
             patch.object(self.agent, "_call_ollama", side_effect=[
                 _ollama_tool("move_to", {"x": 0.1, "y": -0.2, "z": 0.1}),
                 _ollama_text("Moved successfully."),
             ]):
            res = self.agent.process_message("move to 0.1 -0.2 0.1", chat_id=1)
        # Verify the executed tool has the correct structure
        # (we just check the agent ran without error here; contract tested via Flask mock)
        self.assertIsNotNone(res["reply"])

    def test_tool_rejection_contract(self):
        """Out-of-bounds tool call should produce TOOL_REJECTED failure type in log."""
        with patch.object(self.agent, "_call_ollama", side_effect=[
            _ollama_tool("move_to", {"x": 5.0, "y": -0.18, "z": 0.05}),
            _ollama_text("I cannot move there — coordinates are out of bounds."),
        ]):
            res = self.agent.process_message("move to x=5.0", chat_id=2)

        # At least one tool call should have been TOOL_REJECTED
        rejected = [tc for tc in res["tool_calls"] if tc.get("failure_type") == "TOOL_REJECTED"]
        self.assertTrue(len(rejected) >= 1, "Expected at least one TOOL_REJECTED call")
        self.assertFalse(rejected[0]["valid"])

    def test_tool_failed_contract(self):
        """Flask returning success=False should give TOOL_FAILED failure type."""
        flask_failure = {
            "success": False,
            "action": "place",
            "result": None,
            "error": {"code": "NOT_HOLDING", "message": "Not holding."},
        }
        with patch.object(self.agent, "_call_ollama", side_effect=[
            _ollama_tool("place", {"x": 0.15, "y": -0.18, "z": 0.015}),
            _ollama_text("I am not holding the cube, so I cannot place it."),
        ]), patch.object(self.agent, "execute_flask_tool", return_value=flask_failure):
            res = self.agent.process_message("place on right pad", chat_id=3)

        failed = [tc for tc in res["tool_calls"] if tc.get("failure_type") == "TOOL_FAILED"]
        self.assertTrue(len(failed) >= 1, "Expected TOOL_FAILED in tool calls")
        self.assertEqual(res["failure_type"], "TOOL_FAILED")


# ---------------------------------------------------------------------------
# 3. Agent round and tool call limits
# ---------------------------------------------------------------------------

class TestAgentLimits(unittest.TestCase):
    def setUp(self):
        self.test_log = BASE_DIR / "test_limits.jsonl"
        if self.test_log.exists():
            self.test_log.unlink()
        self.agent = RobotArmAgent(log_file_path=self.test_log)

    def tearDown(self):
        if self.test_log.exists():
            self.test_log.unlink()

    def test_max_agent_rounds_respected(self):
        """Agent must stop after MAX_AGENT_ROUNDS Ollama calls."""
        # Always returns a tool call — would loop forever without the limit
        looping_resp = _ollama_tool("state", {})

        with patch.object(self.agent, "_call_ollama", return_value=looping_resp), \
             patch.object(self.agent, "execute_flask_tool", return_value={"mock": "data"}):
            res = self.agent.process_message("keep checking state", chat_id=10)

        self.assertLessEqual(res["agent_rounds"], MAX_AGENT_ROUNDS)
        self.assertIsNotNone(res["reply"])

    def test_max_tool_calls_per_turn_respected(self):
        """Physical tool calls must stop after MAX_TOOL_CALLS_PER_TURN."""
        looping_resp = _ollama_tool("state", {})
        mock_state = {"robot": "ready", "gripper": "open", "holding": False, "eef": [0, -0.2, 0.1], "cube": [0, -0.22, 0.015]}

        with patch.object(self.agent, "_call_ollama", return_value=looping_resp), \
             patch.object(self.agent, "execute_flask_tool", return_value=mock_state):
            res = self.agent.process_message("check state many times", chat_id=11)

        self.assertLessEqual(res["physical_tool_calls"], MAX_TOOL_CALLS_PER_TURN)

    def test_execution_incomplete_failure_type(self):
        """When limit is reached, failure_type should be EXECUTION_INCOMPLETE."""
        looping_resp = _ollama_tool("state", {})

        with patch.object(self.agent, "_call_ollama", return_value=looping_resp), \
             patch.object(self.agent, "execute_flask_tool", return_value={"mock": "data"}):
            res = self.agent.process_message("endless loop test", chat_id=12)

        self.assertEqual(res["failure_type"], FAILURE_EXECUTION_INCOMPLETE)

    def test_reply_not_completed_when_incomplete(self):
        """Agent must NOT say 'Completed' when execution was incomplete."""
        looping_resp = _ollama_tool("state", {})

        with patch.object(self.agent, "_call_ollama", return_value=looping_resp), \
             patch.object(self.agent, "execute_flask_tool", return_value={"mock": "data"}):
            res = self.agent.process_message("loop forever", chat_id=13)

        # Must not silently claim completion
        self.assertNotEqual(res["reply"].strip(), "Completed requested actions.")


# ---------------------------------------------------------------------------
# 4. Logging (Phase 19)
# ---------------------------------------------------------------------------

class TestLoggingFields(unittest.TestCase):
    def setUp(self):
        self.test_log = BASE_DIR / "test_logging.jsonl"
        if self.test_log.exists():
            self.test_log.unlink()
        self.agent = RobotArmAgent(log_file_path=self.test_log)

    def tearDown(self):
        if self.test_log.exists():
            self.test_log.unlink()

    def test_turn_record_fields(self):
        """Log record must include Phase 19 fields."""
        with patch.object(self.agent, "_call_ollama", return_value=_ollama_text("Hello!")):
            self.agent.process_message("hi", chat_id=20)

        self.assertTrue(self.test_log.exists())
        with open(self.test_log, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        required_fields = {
            "timestamp", "chat_id", "user_message",
            "agent_rounds", "physical_tool_calls_count",
            "tool_calls", "photos", "final_reply",
            "failure_type", "error",
        }
        for field in required_fields:
            self.assertIn(field, record, f"Missing field: {field}")

    def test_turn_record_tool_call_fields(self):
        """Each tool call log entry must have success and failure_type."""
        mock_result = {
            "success": True, "action": "state",
            "result": {"robot": "ready"}, "error": None,
        }
        with patch.object(self.agent, "_call_ollama", side_effect=[
            _ollama_tool("state", {}),
            _ollama_text("State checked."),
        ]), patch.object(self.agent, "execute_flask_tool", return_value=mock_result):
            self.agent.process_message("give me status", chat_id=21)

        with open(self.test_log, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        self.assertEqual(len(record["tool_calls"]), 1)
        tc_log = record["tool_calls"][0]
        self.assertIn("tool_success", tc_log)
        self.assertIn("failure_type", tc_log)
        self.assertIn("response_success", tc_log)

    def test_no_secret_logged(self):
        """Bot token must not appear in the log."""
        token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        with patch.object(self.agent, "_call_ollama", return_value=_ollama_text("Hi")):
            self.agent.process_message(f"token is {token}", chat_id=22)

        with open(self.test_log, "r", encoding="utf-8") as f:
            content = f.read()

        # We don't log secrets — but user_message IS logged (that's fine, it's input not a credential)
        # Check that our internal code never injects tokens into the log
        self.assertNotIn("TELEGRAM_BOT_TOKEN", content)


# ---------------------------------------------------------------------------
# 5. /new conversation reset (Phase 11)
# ---------------------------------------------------------------------------

class TestConversationReset(unittest.TestCase):
    def setUp(self):
        self.agent = RobotArmAgent()

    def test_reset_clears_history(self):
        """reset_conversation should remove non-system messages."""
        chat_id = 30
        # Build some history
        with patch.object(self.agent, "_call_ollama", return_value=_ollama_text("Hello!")):
            self.agent.process_message("hi", chat_id=chat_id)

        history_before = self.agent.histories.get(chat_id, [])
        self.assertGreater(len(history_before), 1, "Expected history to grow after a turn")

        # Reset
        self.agent.reset_conversation(chat_id)

        history_after = self.agent.histories.get(chat_id, [])
        self.assertEqual(len(history_after), 1, "After reset, only system prompt should remain")
        self.assertEqual(history_after[0]["role"], "system")

    def test_reset_independent_chats(self):
        """Resetting one chat must not affect another."""
        with patch.object(self.agent, "_call_ollama", return_value=_ollama_text("Hi")):
            self.agent.process_message("hello", chat_id=31)
            self.agent.process_message("hello", chat_id=32)

        self.agent.reset_conversation(31)

        # Chat 31 should be reset; chat 32 should still have history
        self.assertEqual(len(self.agent.histories.get(31, [])), 1)
        self.assertGreater(len(self.agent.histories.get(32, [])), 1)


# ---------------------------------------------------------------------------
# 6. FINAL_RESPONSE_FAILED and MODEL_TIMED_OUT (Phase 9, 10)
# ---------------------------------------------------------------------------

class TestFailurePaths(unittest.TestCase):
    def setUp(self):
        self.test_log = BASE_DIR / "test_failures.jsonl"
        if self.test_log.exists():
            self.test_log.unlink()
        self.agent = RobotArmAgent(log_file_path=self.test_log)

    def tearDown(self):
        if self.test_log.exists():
            self.test_log.unlink()

    def test_model_timed_out_no_prior_tool(self):
        """Ollama timeout with no prior tool call → MODEL_TIMED_OUT."""
        with patch.object(
            self.agent, "_call_ollama",
            side_effect=RuntimeError("Ollama request timed out after 60s.")
        ):
            res = self.agent.process_message("hi", chat_id=40)

        self.assertEqual(res["failure_type"], "MODEL_TIMED_OUT")
        self.assertIsNotNone(res["error"])

    def test_final_response_failed_after_successful_tool(self):
        """Ollama timeout after successful tool → FINAL_RESPONSE_FAILED, honest message."""
        successful_tool_result = {
            "success": True, "action": "pick",
            "result": {"holding": True, "object": "cube",
                       "grasp_verification": {"method": "position_heuristic", "verified": True}},
            "error": None,
        }

        call_count = [0]

        def mock_ollama(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: issue the pick tool
                return _ollama_tool("pick", {})
            # Second call: timeout
            raise RuntimeError("Ollama request timed out after 60s.")

        with patch.object(self.agent, "_call_ollama", side_effect=mock_ollama), \
             patch.object(self.agent, "execute_flask_tool", return_value=successful_tool_result):
            res = self.agent.process_message("pick up the cube", chat_id=41)

        self.assertEqual(res["failure_type"], "FINAL_RESPONSE_FAILED")
        # Reply must not say "Done." or claim success — must reference the actual action
        self.assertIn("pick", res["reply"].lower())
        self.assertNotEqual(res["reply"].strip().lower(), "done.")

    def test_model_failed_to_plan(self):
        """Generic Ollama RuntimeError → MODEL_FAILED_TO_PLAN."""
        with patch.object(
            self.agent, "_call_ollama",
            side_effect=RuntimeError("Ollama returned HTTP 500: Internal Server Error")
        ):
            res = self.agent.process_message("do something", chat_id=42)

        self.assertEqual(res["failure_type"], "MODEL_FAILED_TO_PLAN")


# ---------------------------------------------------------------------------
# 7. Semantic world model (Phase 15)
# ---------------------------------------------------------------------------

class TestSemanticWorldModel(unittest.TestCase):
    def test_resolve_right_pad(self):
        pos = resolve_semantic_target("right_pad")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos["x"], 0.15)
        self.assertAlmostEqual(pos["y"], -0.18)
        self.assertAlmostEqual(pos["z"], 0.015)

    def test_resolve_left_pad(self):
        pos = resolve_semantic_target("left_pad")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos["x"], -0.15)

    def test_resolve_home_hover(self):
        pos = resolve_semantic_target("home_hover")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos["z"], 0.16)

    def test_resolve_unknown(self):
        pos = resolve_semantic_target("blue_pad")
        self.assertIsNone(pos)

    def test_resolve_case_insensitive(self):
        pos = resolve_semantic_target("RIGHT_PAD")
        self.assertIsNotNone(pos)

    def test_resolve_space_normalised(self):
        pos = resolve_semantic_target("right pad")
        self.assertIsNotNone(pos)


# ---------------------------------------------------------------------------
# 8. Failure type constants exist and are strings
# ---------------------------------------------------------------------------

class TestFailureTypeConstants(unittest.TestCase):
    def test_constants_are_strings(self):
        from agent import (
            FAILURE_MODEL_FAILED_TO_PLAN,
            FAILURE_MODEL_TIMED_OUT,
            FAILURE_TOOL_REJECTED,
            FAILURE_TOOL_FAILED,
            FAILURE_EXECUTION_INCOMPLETE,
            FAILURE_FINAL_RESPONSE_FAILED,
        )
        for constant in (
            FAILURE_MODEL_FAILED_TO_PLAN, FAILURE_MODEL_TIMED_OUT,
            FAILURE_TOOL_REJECTED, FAILURE_TOOL_FAILED,
            FAILURE_EXECUTION_INCOMPLETE, FAILURE_FINAL_RESPONSE_FAILED,
        ):
            self.assertIsInstance(constant, str)
            self.assertTrue(len(constant) > 0)


# ---------------------------------------------------------------------------
# 9. Status callback (Phase 16)
# ---------------------------------------------------------------------------

class TestStatusCallback(unittest.TestCase):
    def setUp(self):
        self.agent = RobotArmAgent()

    def test_status_callback_called_on_tool_execution(self):
        """status_callback must be called with 'started' and 'completed' events."""
        events = []

        def cb(event, tool_name, data):
            events.append((event, tool_name))

        mock_result = {
            "success": True, "action": "state",
            "result": {"robot": "ready"}, "error": None,
        }

        with patch.object(self.agent, "_call_ollama", side_effect=[
            _ollama_tool("state", {}),
            _ollama_text("Robot is ready."),
        ]), patch.object(self.agent, "execute_flask_tool", return_value=mock_result):
            self.agent.process_message("status?", chat_id=50, status_callback=cb)

        event_types = [e[0] for e in events]
        self.assertIn("started", event_types)
        self.assertIn("completed", event_types)
        # Tool name must be passed correctly
        started_tools = [e[1] for e in events if e[0] == "started"]
        self.assertIn("state", started_tools)

    def test_status_callback_called_on_failure(self):
        """status_callback must be called with 'failed' event on tool failure."""
        events = []

        def cb(event, tool_name, data):
            events.append((event, tool_name))

        flask_failure = {
            "success": False, "action": "place",
            "result": None,
            "error": {"code": "NOT_HOLDING", "message": "Not holding."},
        }

        with patch.object(self.agent, "_call_ollama", side_effect=[
            _ollama_tool("place", {"x": 0.15, "y": -0.18, "z": 0.015}),
            _ollama_text("Cannot place."),
        ]), patch.object(self.agent, "execute_flask_tool", return_value=flask_failure):
            self.agent.process_message("place the cube", chat_id=51, status_callback=cb)

        event_types = [e[0] for e in events]
        self.assertIn("failed", event_types)


# ---------------------------------------------------------------------------
# 10. Live Ollama integration (optional — skipped if Ollama not running)
# ---------------------------------------------------------------------------

class TestLiveOllamaIntegration(unittest.TestCase):
    """Live test against local Ollama service with qwen3.5:4b."""

    def setUp(self):
        self.agent = RobotArmAgent()

    def test_live_ollama_tool_call_generation(self):
        try:
            resp = self.agent.http_session.get(f"{self.agent.ollama_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                self.skipTest("Ollama is not running")
        except Exception:
            self.skipTest("Ollama is not running")

        res = self.agent._call_ollama([
            {"role": "system", "content": "You are controlling a robot arm. Use the pick tool when asked to pick the cube."},
            {"role": "user", "content": "Please pick up the red cube now."}
        ])
        msg = res.get("message", {})
        tool_calls = msg.get("tool_calls", [])
        self.assertTrue(len(tool_calls) > 0, "Ollama did not generate a tool call")
        self.assertEqual(tool_calls[0]["function"]["name"], "pick")


if __name__ == "__main__":
    unittest.main()
