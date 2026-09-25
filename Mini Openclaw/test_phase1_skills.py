"""
test_phase1_skills.py — Unit and integration tests for Phase 1:
1. Truthful physical state (holding, held_object, multi-object resolution).
2. Deterministic high-level skill layer (go_home, Scenario A/B/C).
3. Failure propagation and step-by-step verification.
4. Independent testability without Flask, MuJoCo viewer, Telegram, or Ollama.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root and Mini Openclaw are on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import skills
from robot_api import RobotAPI
from agent import RobotArmAgent, FAILURE_TOOL_FAILED


class MockRobot:
    """Mock robot to test skills without MuJoCo or Flask."""

    def __init__(self):
        self.pick_result = {
            "success": True,
            "action": "pick",
            "result": {"holding": True, "object": "cube"},
            "error": None,
        }
        self.place_result = {
            "success": True,
            "action": "place",
            "result": {"released": True, "position": [0.15, -0.18, 0.015]},
            "error": None,
        }
        self.move_to_result = {
            "success": True,
            "action": "move_to",
            "result": {"position_error": 0.001},
            "error": None,
        }
        self.reset_home_result = {
            "success": True,
            "action": "reset_home",
            "result": {"eef": [0.0, -0.347, 0.202]},
            "error": None,
        }
        self._holding = False
        self._held_object = None
        self.force_holding_after_pick = None
        self.calls = []

    def get_state(self):
        self.calls.append("get_state")
        return {
            "holding": self._holding,
            "held_object": self._held_object,
        }

    def pick(self, target="cube"):
        self.calls.append(f"pick({target})")
        if self.force_holding_after_pick is not None:
            self._holding = self.force_holding_after_pick
        elif self.pick_result.get("success"):
            self._holding = True
            self._held_object = target
        return self.pick_result

    def place(self, x, y, z):
        self.calls.append(f"place({x}, {y}, {z})")
        if self.place_result.get("success"):
            self._holding = False
            self._held_object = None
        return self.place_result

    def move_to(self, x, y, z, steps=80):
        self.calls.append(f"move_to({x}, {y}, {z})")
        return self.move_to_result

    def reset_home(self, steps=80):
        self.calls.append("reset_home")
        return self.reset_home_result

    def save_camera_image(self, filepath="inspection_high.png"):
        self.calls.append(f"save_camera_image({filepath})")
        return filepath


class TestSkillLayerMocked(unittest.TestCase):
    """Test deterministic skill execution, verification, and abort behavior using mocks."""

    def setUp(self):
        self.robot = MockRobot()

    def test_go_home_success(self):
        res = skills.go_home(self.robot)
        self.assertTrue(res["success"])
        self.assertEqual(res["action"], "go_home")
        self.assertIsNone(res["error"])
        self.assertTrue(res["result"]["homed"])
        self.assertIn("reset_home", self.robot.calls)

    def test_go_home_failure(self):
        self.robot.reset_home_result = {
            "success": False,
            "action": "reset_home",
            "result": None,
            "error": {"code": "MOTOR_STALL", "message": "Arm jammed."},
        }
        res = skills.go_home(self.robot)
        self.assertFalse(res["success"])
        self.assertEqual(res["action"], "go_home")
        self.assertEqual(res["error"]["code"], "MOTOR_STALL")

    def test_scenario_a_success(self):
        res = skills.scenario_a(self.robot, target="cube_1")
        self.assertTrue(res["success"])
        self.assertEqual(res["action"], "scenario")
        self.assertIsNone(res["error"])
        self.assertEqual(res["result"]["scenario"], "A")
        self.assertEqual(res["result"]["target"], "cube_1")
        self.assertEqual(
            res["result"]["steps_completed"],
            ["pick", "verify_holding", "place", "verify_released", "go_home"]
        )
        self.assertIn("pick(cube_1)", self.robot.calls)
        self.assertIn("place(0.15, -0.18, 0.015)", self.robot.calls)
        self.assertIn("reset_home", self.robot.calls)

    def test_scenario_a_pick_failed_aborts_immediately(self):
        self.robot.pick_result = {
            "success": False,
            "action": "pick",
            "result": None,
            "error": {"code": "EXECUTION_FAILED", "message": "Grasp slipped."},
        }
        res = skills.scenario_a(self.robot, target="cube_2")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "EXECUTION_FAILED")
        self.assertEqual(res["result"]["step_failed"], "pick")
        # Critical test: place MUST NOT be called after pick failed
        place_calls = [c for c in self.robot.calls if c.startswith("place")]
        self.assertEqual(len(place_calls), 0, "Place was called despite failed pick!")

    def test_scenario_a_holding_verification_failure_aborts(self):
        # Pick reports success, but robot state reveals it is NOT holding
        self.robot.pick_result = {"success": True, "action": "pick", "result": {}, "error": None}
        self.robot.force_holding_after_pick = False  # simulated drop
        res = skills.scenario_a(self.robot, target="cube_1")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "VERIFICATION_FAILED")
        self.assertEqual(res["result"]["step_failed"], "verify_holding")
        place_calls = [c for c in self.robot.calls if c.startswith("place")]
        self.assertEqual(len(place_calls), 0, "Place was called despite verification failure!")

    def test_scenario_a_place_failed_aborts(self):
        self.robot.place_result = {
            "success": False,
            "action": "place",
            "result": None,
            "error": {"code": "COLLISION", "message": "Obstacle at drop pad."},
        }
        res = skills.scenario_a(self.robot, target="cube")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "COLLISION")
        self.assertEqual(res["result"]["step_failed"], "place")

    def test_scenario_b_success(self):
        res = skills.scenario_b(self.robot, target="cube_2")
        self.assertTrue(res["success"])
        self.assertEqual(res["result"]["scenario"], "B")
        self.assertEqual(res["result"]["target"], "cube_2")
        self.assertIn("place(-0.15, -0.18, 0.015)", self.robot.calls)

    def test_scenario_b_pick_failed_does_not_call_place(self):
        self.robot.pick_result = {
            "success": False,
            "action": "pick",
            "result": None,
            "error": {"code": "OBJECT_NOT_FOUND", "message": "Cube missing."},
        }
        res = skills.scenario_b(self.robot, target="cube_missing")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "OBJECT_NOT_FOUND")
        place_calls = [c for c in self.robot.calls if c.startswith("place")]
        self.assertEqual(len(place_calls), 0)

    def test_scenario_c_success(self):
        res = skills.scenario_c(self.robot)
        self.assertTrue(res["success"])
        self.assertEqual(res["result"]["scenario"], "C")
        self.assertIn("move_to(0.0, -0.2, 0.16)", self.robot.calls)
        self.assertIn("move_to(-0.12, -0.22, 0.12)", self.robot.calls)
        self.assertIn("move_to(0.12, -0.22, 0.12)", self.robot.calls)
        self.assertIn("reset_home", self.robot.calls)

    def test_scenario_c_move_failure_aborts(self):
        self.robot.move_to_result = {
            "success": False,
            "action": "move_to",
            "result": None,
            "error": {"code": "IK_FAILED", "message": "Unreachable target."},
        }
        res = skills.scenario_c(self.robot)
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "IK_FAILED")
        self.assertEqual(res["result"]["step_failed"], "move_high")

    def test_run_scenario_dispatcher(self):
        # Test valid names case-insensitive
        self.assertTrue(skills.run_scenario(self.robot, "A")["success"])
        self.assertTrue(skills.run_scenario(self.robot, "b")["success"])
        self.assertTrue(skills.run_scenario(self.robot, "C")["success"])

        # Test invalid name
        invalid_res = skills.run_scenario(self.robot, "Z")
        self.assertFalse(invalid_res["success"])
        self.assertEqual(invalid_res["error"]["code"], "INVALID_ARGUMENT")


class TestPhysicalStateAndMultiObject(unittest.TestCase):
    """Test physical simulation truthfulness and multi-object representation using headless MuJoCo."""

    @classmethod
    def setUpClass(cls):
        cls.robot = RobotAPI(render=False)

    @classmethod
    def tearDownClass(cls):
        cls.robot.close()

    def test_initial_state_not_holding(self):
        st = self.robot.get_state()
        self.assertFalse(st["holding"])
        self.assertIsNone(st["held_object"])
        self.assertFalse(st["holding_cube"])

    def test_registered_objects_present(self):
        st = self.robot.get_state()
        self.assertIn("objects", st)
        objects = st["objects"]
        expected_keys = {"cube_1", "cube_2", "cube_3", "sphere"}
        self.assertTrue(expected_keys.issubset(set(objects.keys())),
                        f"Expected {expected_keys}, got {set(objects.keys())}")

        for name, o_data in objects.items():
            pos = o_data["position"]
            self.assertIn("x", pos)
            self.assertIn("y", pos)
            self.assertIn("z", pos)
            self.assertTrue(o_data["in_workspace"], f"Object {name} reported outside workspace: {pos}")
            self.assertFalse(o_data["holding"])

    def test_semantic_state_multi_object_structure(self):
        sem = self.robot.get_semantic_state()
        self.assertIn("objects", sem)
        self.assertIn("cube_1", sem["objects"])
        self.assertIn("cube_2", sem["objects"])
        self.assertIn("cube_3", sem["objects"])
        self.assertIn("sphere", sem["objects"])
        # Backward compatibility keys preserved
        self.assertIn("cube", sem)
        self.assertIn("sphere", sem)
        self.assertEqual(len(sem["cube"]), 3)
        self.assertEqual(len(sem["sphere"]), 3)

    def test_resolve_object_target_synonyms(self):
        # Red cube
        name, bid = self.robot._resolve_object_target("cube")
        self.assertEqual(name, "cube_1")
        self.assertNotEqual(bid, -1)

        name, bid = self.robot._resolve_object_target("red_cube")
        self.assertEqual(name, "cube_1")

        # Green cube
        name, bid = self.robot._resolve_object_target("cube_2")
        self.assertEqual(name, "cube_2")
        self.assertNotEqual(bid, -1)

        name, bid = self.robot._resolve_object_target("green_cube")
        self.assertEqual(name, "cube_2")

        # Yellow cube
        name, bid = self.robot._resolve_object_target("cube_3")
        self.assertEqual(name, "cube_3")
        self.assertNotEqual(bid, -1)

        name, bid = self.robot._resolve_object_target("yellow_cube")
        self.assertEqual(name, "cube_3")

        # Sphere
        name, bid = self.robot._resolve_object_target("sphere")
        self.assertEqual(name, "sphere")
        self.assertNotEqual(bid, -1)

        # Unknown object
        name, bid = self.robot._resolve_object_target("non_existent_pyramid")
        self.assertIsNone(name)
        self.assertEqual(bid, -1)

    def test_pick_unknown_object_fails_safely(self):
        res = self.robot.pick(target="phantom_object")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "OBJECT_NOT_FOUND")
        self.assertFalse(self.robot._holding)
        self.assertIsNone(self.robot._held_object)

    def test_place_without_holding_fails_safely(self):
        # Robot is not holding anything
        self.robot._holding = False
        self.robot._held_object = None
        res = self.robot.place(0.15, -0.18, 0.015)
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "NOT_HOLDING")

    def test_robot_api_scenario_method(self):
        # RobotAPI has a scenario method that returns structured contract
        res = self.robot.scenario("UNKNOWN_SCENARIO")
        self.assertFalse(res["success"])
        self.assertEqual(res["action"], "scenario")
        self.assertEqual(res["error"]["code"], "INVALID_ARGUMENT")

    def test_live_scenario_c(self):
        res = self.robot.scenario("C")
        self.assertTrue(res["success"])
        self.assertEqual(res["result"]["scenario"], "C")


class TestAgentSkillIntegration(unittest.TestCase):
    """Test agent-side validation and execution paths for go_home and multi-cube targets."""

    def setUp(self):
        self.agent = RobotArmAgent()

    def test_validation_go_home_valid(self):
        valid, err = self.agent.validate_tool_call("go_home", {})
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_validation_pick_multi_cubes(self):
        valid, err = self.agent.validate_tool_call("pick", {"target": "cube_1"})
        self.assertTrue(valid)
        valid, err = self.agent.validate_tool_call("pick", {"target": "cube_2"})
        self.assertTrue(valid)
        valid, err = self.agent.validate_tool_call("pick", {"target": "green_cube"})
        self.assertTrue(valid)
        valid, err = self.agent.validate_tool_call("pick", {"target": "cube_3"})
        self.assertTrue(valid)
        valid, err = self.agent.validate_tool_call("pick", {"target": "yellow_box"})
        self.assertTrue(valid)

    def test_scenario_failure_propagation_to_agent(self):
        # Mock execute_flask_tool returning structured failure
        self.agent.execute_flask_tool = MagicMock(return_value={
            "success": False,
            "action": "scenario",
            "result": {"scenario": "A", "step_failed": "pick"},
            "error": {"code": "EXECUTION_FAILED", "message": "Grasp failed."},
        })
        self.agent._call_ollama = MagicMock(side_effect=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "scenario", "arguments": {"name": "A"}},
                    }],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Scenario A aborted because grasp failed.",
                }
            },
        ])

        res = self.agent.process_message("run scenario A")
        self.assertEqual(res["failure_type"], FAILURE_TOOL_FAILED)
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertFalse(res["tool_calls"][0]["tool_success"])
        self.assertEqual(res["tool_calls"][0]["failure_type"], FAILURE_TOOL_FAILED)


if __name__ == "__main__":
    unittest.main()
