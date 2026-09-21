"""
test_agent.py — Test suite for the Mini Openclaw Robot Arm Agent.

Verifies:
1. Schema integrity for all 9 endpoints
2. Strict coordinate envelope and type validation (rejects without clamping)
3. Multi-step loop bounds and history management
4. JSONL turn logging
5. Live Ollama tool calling with qwen3.5:4b
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent import RobotArmAgent
from config import BASE_DIR, LOG_FILE_PATH, REACHABLE_ENVELOPE, TOOLS_SCHEMA_PATH


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

    def test_validation_gripper(self):
        # Valid values
        self.assertTrue(self.agent.validate_tool_call("gripper", {"value": 0.0})[0])
        self.assertTrue(self.agent.validate_tool_call("gripper", {"value": 1.0})[0])
        self.assertTrue(self.agent.validate_tool_call("gripper", {"value": 0.5})[0])

        # Invalid values
        self.assertFalse(self.agent.validate_tool_call("gripper", {"value": 1.5})[0])
        self.assertFalse(self.agent.validate_tool_call("gripper", {"value": -0.1})[0])
        self.assertFalse(self.agent.validate_tool_call("gripper", {"value": "open"})[0])

    def test_validation_scenario(self):
        self.assertTrue(self.agent.validate_tool_call("scenario", {"name": "A"})[0])
        self.assertTrue(self.agent.validate_tool_call("scenario", {"name": "b"})[0])
        self.assertTrue(self.agent.validate_tool_call("scenario", {"name": "C"})[0])
        self.assertFalse(self.agent.validate_tool_call("scenario", {"name": "D"})[0])

    def test_validation_zero_arg_tools(self):
        for tool in ("pick", "state", "camera", "reset_home", "collisions"):
            valid, err = self.agent.validate_tool_call(tool, {})
            self.assertTrue(valid, f"Failed for {tool}: {err}")


class TestAgentLoopAndLogging(unittest.TestCase):
    def setUp(self):
        self.test_log = BASE_DIR / "test_turns.jsonl"
        if self.test_log.exists():
            self.test_log.unlink()
        self.agent = RobotArmAgent(log_file_path=self.test_log)

    def tearDown(self):
        if self.test_log.exists():
            self.test_log.unlink()

    def test_logging_turn_record(self):
        # Process a message with mocked Ollama and Flask
        mock_ollama_resp = {
            "message": {
                "role": "assistant",
                "content": "I have moved the arm to the home rest position.",
                "tool_calls": []
            }
        }
        with patch.object(self.agent, "_call_ollama", return_value=mock_ollama_resp):
            res = self.agent.process_message("Reset home", chat_id=123)
            self.assertEqual(res["reply"], "I have moved the arm to the home rest position.")

        # Check JSONL output
        self.assertTrue(self.test_log.exists())
        with open(self.test_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["chat_id"], 123)
        self.assertEqual(record["user_message"], "Reset home")
        self.assertEqual(record["final_reply"], "I have moved the arm to the home rest position.")

    def test_bounded_multi_step_loop(self):
        """Verify the loop does not exceed MAX_TOOL_STEPS."""
        # Always return a tool call to simulate a runaway loop
        mock_looping_resp = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_loop",
                        "type": "function",
                        "function": {"name": "state", "arguments": {}}
                    }
                ]
            }
        }
        with patch.object(self.agent, "_call_ollama", return_value=mock_looping_resp), \
             patch.object(self.agent, "execute_flask_tool", return_value={"mock": "data"}):
            res = self.agent.process_message("Keep checking state", chat_id=456)
            self.assertIsNotNone(res["reply"])
            # Should have stopped after MAX_TOOL_STEPS (4)
            self.assertLessEqual(len(res["tool_calls"]), 4)


class TestLiveOllamaIntegration(unittest.TestCase):
    """Live test against local Ollama service with qwen3.5:4b."""

    def setUp(self):
        self.agent = RobotArmAgent()

    def test_live_ollama_tool_call_generation(self):
        try:
            # Check if Ollama is reachable
            resp = self.agent.http_session.get(f"{self.agent.ollama_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                self.skipTest("Ollama is not running")
        except Exception:
            self.skipTest("Ollama is not running")

        # Test prompt requiring pick
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
