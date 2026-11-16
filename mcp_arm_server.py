"""mcp_arm_server.py - stdio MCP bridge to the XavierClaw Flask arm API.

Thin HTTP client only: it must never import RobotAPI, otherwise a second
MuJoCo viewer would be created. Start server.py (or launcher.py) first.

Run standalone:
    python mcp_arm_server.py
"""

import os
import sys
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

# Reuse the canonical safety limits used by the Python agent.  The MCP path is
# an independent entry point, so it must enforce the same gates itself.
_MINI_DIR = Path(__file__).resolve().parent / "Mini Openclaw"
if str(_MINI_DIR) not in sys.path:
    sys.path.insert(0, str(_MINI_DIR))

try:
    from config import GRIPPER_RANGE, REACHABLE_ENVELOPE, VALID_SCENARIOS
except Exception:  # pragma: no cover - fallback mirrors Mini Openclaw/config.py
    REACHABLE_ENVELOPE = {
        "x": (-0.30, 0.30),
        "y": (-0.35, -0.05),
        "z": (0.00, 0.35),
    }
    GRIPPER_RANGE = (0.0, 1.0)
    VALID_SCENARIOS = {"A", "B", "C"}

BASE = os.getenv("ARM_API_URL", "http://127.0.0.1:8765").rstrip("/")
try:
    _configured_timeout = int(os.getenv("ARM_TIMEOUT_SECONDS", "120"))
except ValueError:
    _configured_timeout = 120
# Pick/place can take 30-40 s; never allow a shorter Flask timeout than 60 s.
TIMEOUT = max(60, _configured_timeout)

_session = requests.Session()
mcp = FastMCP("arm")


def _get(path):
    r = _session.get(f"{BASE}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path, payload=None):
    r = _session.post(f"{BASE}{path}", json=payload or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _reject(tool_name, message):
    """Mirror agent.py's TOOL_REJECTED envelope; never clamp the value."""
    return {
        "success": False,
        "action": tool_name,
        "result": None,
        "error": {
            "code": "TOOL_REJECTED",
            "message": message,
        },
        "clarification_needed": True,
    }


def _validate_position(tool_name, x, y, z):
    try:
        values = {"x": float(x), "y": float(y), "z": float(z)}
    except (TypeError, ValueError):
        return None, _reject(
            tool_name,
            "Coordinates x, y, z must be valid numbers.",
        )

    for axis, value in values.items():
        low, high = REACHABLE_ENVELOPE[axis]
        if not (low <= value <= high):
            return None, _reject(
                tool_name,
                f"Target {axis.upper()} coordinate {value:+.3f}m is outside "
                f"the reachable envelope [{low:.2f}, {high:.2f}]m. "
                "Action rejected.",
            )
    return values, None


def _normalise_pick_target(target):
    raw_target = str(target).strip().lower()
    clean_target = "_".join(raw_target.replace("-", " ").split())

    valid_cubes = {
        "cube", "cube_1", "cube_2", "cube_3",
        "red_cube", "green_cube", "yellow_cube",
        "red_box", "green_box", "yellow_box", "box",
    }
    valid_spheres = {
        "sphere", "sphere_1", "blue_sphere", "ball", "blue_ball",
    }
    if clean_target not in valid_cubes and clean_target not in valid_spheres:
        return None
    if clean_target in valid_spheres:
        return "sphere"
    if clean_target in {"cube_2", "green_cube", "green_box"}:
        return "cube_2"
    if clean_target in {"cube_3", "yellow_cube", "yellow_box"}:
        return "cube_3"
    return "cube"


@mcp.tool()
def state() -> dict:
    """Get concise robot, gripper, and object state."""
    return _get("/state")


@mcp.tool()
def state_full() -> dict:
    """Get full MuJoCo telemetry for debugging."""
    return _get("/state/full")


@mcp.tool()
def move_to(x: float, y: float, z: float) -> dict:
    """Move the end-effector to absolute x, y, z in meters.

    Reachable envelope: x -0.30..0.30, y -0.35..-0.05, z 0.00..0.35.
    """
    values, error = _validate_position("move_to", x, y, z)
    if error:
        return error
    return _post("/move_to", values)


@mcp.tool()
def pick(target: str = "cube") -> dict:
    """Pick cube, cube_2, cube_3, or sphere with the deterministic side approach."""
    normalised = _normalise_pick_target(target)
    if normalised is None:
        return _reject(
            "pick",
            f"Invalid pick target '{target}'. Must be 'cube' or 'sphere'.",
        )
    return _post("/pick", {"target": normalised})


@mcp.tool()
def place(x: float, y: float, z: float) -> dict:
    """Place the held object at absolute x, y, z and release it."""
    values, error = _validate_position("place", x, y, z)
    if error:
        return error
    return _post("/place", values)


@mcp.tool()
def gripper(value: float) -> dict:
    """Set jaw openness: 0.0 = fully open, 1.0 = fully closed."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return _reject("gripper", "Gripper value must be a number between 0.0 and 1.0.")
    low, high = GRIPPER_RANGE
    if not (low <= numeric_value <= high):
        return _reject(
            "gripper",
            f"Gripper value {numeric_value} is outside valid range "
            f"[{low}, {high}]. Action rejected.",
        )
    return _post("/gripper", {"value": numeric_value})


@mcp.tool()
def reset_home() -> dict:
    """Return the arm to its home resting pose."""
    return _post("/reset_home")


@mcp.tool()
def scenario(name: str, target: str = "cube") -> dict:
    """Run scenario A (right pad), B (left pad), or C (inspection wave)."""
    scenario_name = str(name).strip().upper()
    if scenario_name not in VALID_SCENARIOS:
        return _reject(
            "scenario",
            f"Scenario '{name}' is unknown. Must be one of "
            f"{sorted(VALID_SCENARIOS)}.",
        )
    normalised_target = _normalise_pick_target(target)
    if normalised_target is None:
        return _reject(
            "scenario",
            f"Invalid scenario target '{target}'. Must be 'cube' or 'sphere'.",
        )
    return _post(
        "/scenario",
        {"name": scenario_name, "target": normalised_target},
    )


@mcp.tool()
def camera() -> dict:
    """Capture a wrist-camera image and return the saved PNG path."""
    return _get("/camera")


@mcp.tool()
def collisions() -> dict:
    """List the current physical contact pairs."""
    return _get("/collisions")


if __name__ == "__main__":
    mcp.run()
