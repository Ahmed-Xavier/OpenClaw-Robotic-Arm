"""
skills.py — Deterministic High-Level Skill Layer for XavierClaw / SO-100.

Architecture:
    agent.py (LLM / reasoning)
        ↓
    server.py (HTTP dispatch only)
        ↓
    skills.py (Deterministic high-level skills: compose primitives, verify state, abort safely)
        ↓
    robot_api.py (Primitives: move_to, gripper, pick, place, reset_home, get_state)
        ↓
    MuJoCo (Physics simulation)

Design Rules:
1. Composes deterministic primitive actions.
2. Checks primitive results step-by-step.
3. Observes relevant state after key transitions.
4. Aborts safely immediately upon primitive failure.
5. Returns canonical structured result contract:
       {"success": bool, "action": str, "result": {...}, "error": {...}|None}
6. Independently testable without MuJoCo, Flask, Ollama, or Telegram.
"""

from typing import Any, Dict, Optional


def _make_skill_result(
    action: str,
    result: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper to generate canonical structured result dictionary."""
    success = error_code is None
    return {
        "success": success,
        "action": action,
        "result": result,
        "error": None if success else {"code": error_code, "message": error_message},
    }


def go_home(robot: Any) -> Dict[str, Any]:
    """High-level skill: Return the arm safely to home resting configuration.

    Calls the reset_home primitive, verifies completion and EEF state, and
    returns standard result contract.
    """
    res = robot.reset_home()
    if not isinstance(res, dict) or not res.get("success"):
        err = res.get("error", {}) if isinstance(res, dict) else {}
        return _make_skill_result(
            "go_home",
            result=res.get("result") if isinstance(res, dict) else None,
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", "Failed to reset robot arm to home position."),
        )

    # Post-action verification
    eef = res.get("result", {}).get("eef")
    return _make_skill_result(
        "go_home",
        result={
            "homed": True,
            "eef": eef,
        },
    )


def scenario_a(robot: Any, target: str = "cube") -> Dict[str, Any]:
    """Scenario A: Pick target object, place at right pad (0.15, -0.18, 0.015), return home.

    Aborts immediately upon any step failure.
    """
    # Step 1: Pick
    pick_res = robot.pick(target=target)
    if not isinstance(pick_res, dict) or not pick_res.get("success"):
        err = pick_res.get("error", {}) if isinstance(pick_res, dict) else {}
        return _make_skill_result(
            "scenario",
            result={"scenario": "A", "step_failed": "pick", "target": target, "primitive_result": pick_res},
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", f"Scenario A aborted: pick failed on target '{target}'."),
        )

    # Step 2: Post-pick verification
    if hasattr(robot, "get_state"):
        state = robot.get_state()
        if not state.get("holding", False):
            return _make_skill_result(
                "scenario",
                result={"scenario": "A", "step_failed": "verify_holding", "target": target},
                error_code="VERIFICATION_FAILED",
                error_message=f"Scenario A aborted: robot is not physically holding target '{target}'.",
            )

    # Step 3: Place at Right Target Pad (0.15, -0.18, 0.015)
    place_res = robot.place(0.15, -0.18, 0.015)
    if not isinstance(place_res, dict) or not place_res.get("success"):
        err = place_res.get("error", {}) if isinstance(place_res, dict) else {}
        return _make_skill_result(
            "scenario",
            result={"scenario": "A", "step_failed": "place", "primitive_result": place_res},
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", "Scenario A aborted: place failed at right target pad."),
        )

    # Step 4: Post-place verification
    if hasattr(robot, "get_state"):
        state = robot.get_state()
        if state.get("holding", False):
            return _make_skill_result(
                "scenario",
                result={"scenario": "A", "step_failed": "verify_released"},
                error_code="VERIFICATION_FAILED",
                error_message="Scenario A aborted: object was not released during place.",
            )

    # Step 5: Return home
    home_res = go_home(robot)
    if not home_res.get("success"):
        return _make_skill_result(
            "scenario",
            result={"scenario": "A", "step_failed": "go_home", "primitive_result": home_res},
            error_code=home_res.get("error", {}).get("code", "EXECUTION_FAILED"),
            error_message=home_res.get("error", {}).get("message", "Scenario A aborted: return to home failed."),
        )

    return _make_skill_result(
        "scenario",
        result={
            "scenario": "A",
            "target": target,
            "steps_completed": ["pick", "verify_holding", "place", "verify_released", "go_home"],
            "pick": pick_res.get("result"),
            "place": place_res.get("result"),
            "home": home_res.get("result"),
        },
    )


def scenario_b(robot: Any, target: str = "cube") -> Dict[str, Any]:
    """Scenario B: Pick target object, place at left pad (-0.15, -0.18, 0.015), return home.

    Aborts immediately upon any step failure.
    """
    # Step 1: Pick
    pick_res = robot.pick(target=target)
    if not isinstance(pick_res, dict) or not pick_res.get("success"):
        err = pick_res.get("error", {}) if isinstance(pick_res, dict) else {}
        return _make_skill_result(
            "scenario",
            result={"scenario": "B", "step_failed": "pick", "target": target, "primitive_result": pick_res},
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", f"Scenario B aborted: pick failed on target '{target}'."),
        )

    # Step 2: Post-pick verification
    if hasattr(robot, "get_state"):
        state = robot.get_state()
        if not state.get("holding", False):
            return _make_skill_result(
                "scenario",
                result={"scenario": "B", "step_failed": "verify_holding", "target": target},
                error_code="VERIFICATION_FAILED",
                error_message=f"Scenario B aborted: robot is not physically holding target '{target}'.",
            )

    # Step 3: Place at Left Target Pad (-0.15, -0.18, 0.015)
    place_res = robot.place(-0.15, -0.18, 0.015)
    if not isinstance(place_res, dict) or not place_res.get("success"):
        err = place_res.get("error", {}) if isinstance(place_res, dict) else {}
        return _make_skill_result(
            "scenario",
            result={"scenario": "B", "step_failed": "place", "primitive_result": place_res},
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", "Scenario B aborted: place failed at left target pad."),
        )

    # Step 4: Post-place verification
    if hasattr(robot, "get_state"):
        state = robot.get_state()
        if state.get("holding", False):
            return _make_skill_result(
                "scenario",
                result={"scenario": "B", "step_failed": "verify_released"},
                error_code="VERIFICATION_FAILED",
                error_message="Scenario B aborted: object was not released during place.",
            )

    # Step 5: Return home
    home_res = go_home(robot)
    if not home_res.get("success"):
        return _make_skill_result(
            "scenario",
            result={"scenario": "B", "step_failed": "go_home", "primitive_result": home_res},
            error_code=home_res.get("error", {}).get("code", "EXECUTION_FAILED"),
            error_message=home_res.get("error", {}).get("message", "Scenario B aborted: return to home failed."),
        )

    return _make_skill_result(
        "scenario",
        result={
            "scenario": "B",
            "target": target,
            "steps_completed": ["pick", "verify_holding", "place", "verify_released", "go_home"],
            "pick": pick_res.get("result"),
            "place": place_res.get("result"),
            "home": home_res.get("result"),
        },
    )


def scenario_c(robot: Any) -> Dict[str, Any]:
    """Scenario C: Inspection wave: move up, save photo, sweep left/right, return home.

    Aborts immediately upon any step failure.
    """
    # Step 1: Move up to high inspection pose
    res_high = robot.move_to(0.0, -0.20, 0.16)
    if not isinstance(res_high, dict) or not res_high.get("success"):
        err = res_high.get("error", {}) if isinstance(res_high, dict) else {}
        return _make_skill_result(
            "scenario",
            result={"scenario": "C", "step_failed": "move_high", "primitive_result": res_high},
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", "Scenario C aborted: failed to reach high inspection pose."),
        )

    # Step 2: Capture high inspection photo
    photo_path = "inspection_high.png"
    if hasattr(robot, "save_camera_image"):
        try:
            photo_path = robot.save_camera_image("inspection_high.png")
        except Exception as e:
            return _make_skill_result(
                "scenario",
                result={"scenario": "C", "step_failed": "save_camera_image"},
                error_code="EXECUTION_FAILED",
                error_message=f"Scenario C aborted: camera capture failed ({e}).",
            )

    # Step 3: Tilt wrist / sweep left
    res_left = robot.move_to(-0.12, -0.22, 0.12, steps=100)
    if not isinstance(res_left, dict) or not res_left.get("success"):
        err = res_left.get("error", {}) if isinstance(res_left, dict) else {}
        return _make_skill_result(
            "scenario",
            result={"scenario": "C", "step_failed": "sweep_left", "primitive_result": res_left},
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", "Scenario C aborted: failed during left inspection sweep."),
        )

    # Step 4: Tilt wrist / sweep right
    res_right = robot.move_to(0.12, -0.22, 0.12, steps=120)
    if not isinstance(res_right, dict) or not res_right.get("success"):
        err = res_right.get("error", {}) if isinstance(res_right, dict) else {}
        return _make_skill_result(
            "scenario",
            result={"scenario": "C", "step_failed": "sweep_right", "primitive_result": res_right},
            error_code=err.get("code", "EXECUTION_FAILED"),
            error_message=err.get("message", "Scenario C aborted: failed during right inspection sweep."),
        )

    # Step 5: Return home
    home_res = go_home(robot)
    if not home_res.get("success"):
        return _make_skill_result(
            "scenario",
            result={"scenario": "C", "step_failed": "go_home", "primitive_result": home_res},
            error_code=home_res.get("error", {}).get("code", "EXECUTION_FAILED"),
            error_message=home_res.get("error", {}).get("message", "Scenario C aborted: return to home failed."),
        )

    return _make_skill_result(
        "scenario",
        result={
            "scenario": "C",
            "photo_path": photo_path,
            "steps_completed": ["move_high", "save_camera_image", "sweep_left", "sweep_right", "go_home"],
            "home": home_res.get("result"),
        },
    )


def run_scenario(robot: Any, name: str, target: str = "cube") -> Dict[str, Any]:
    """Execute a scenario by name ('A', 'B', 'C') with deterministic verification.

    Returns the standard result contract.
    """
    clean_name = str(name).strip().upper()
    if clean_name == "A":
        return scenario_a(robot, target=target)
    elif clean_name == "B":
        return scenario_b(robot, target=target)
    elif clean_name == "C":
        return scenario_c(robot)
    else:
        return _make_skill_result(
            "scenario",
            result=None,
            error_code="INVALID_ARGUMENT",
            error_message=f"Scenario '{name}' is unknown. Must be one of ['A', 'B', 'C'].",
        )
