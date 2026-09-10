"""
server.py — Flask REST server for the MuJoCo SO-ARM100 simulation.

Exposes a thin HTTP API over a single global RobotAPI instance so that
external clients (LLM agents, scripts, etc.) can drive the arm without
knowing anything about MuJoCo.

Endpoints
---------
POST /move_to      {"x": float, "y": float, "z": float}
POST /pick         {}
POST /place        {"x": float, "y": float, "z": float}
POST /gripper      {"value": float}
GET  /state
GET  /camera       -> {"path": "<absolute path to saved PNG>"}
POST /reset_home   {}

All successful responses are JSON.  Errors return {"error": "<message>"} with
HTTP 400 so a bad request never crashes the simulation process.

Run
---
    python server.py

The server binds to 127.0.0.1:8765.  debug=False and use_reloader=False are
required: the Werkzeug reloader would fork a second process and instantiate a
second RobotAPI / MuJoCo viewer, which breaks the simulation.
"""

import os
import tempfile

from flask import Flask, jsonify, request

from robot_api import RobotAPI

# ---------------------------------------------------------------------------
# Global robot instance — created exactly once at startup, never per-request.
# ---------------------------------------------------------------------------
robot = RobotAPI(render=True)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ok(result):
    """Return a JSON response from a dict result."""
    return jsonify(result)


def _err(exc, status=400):
    """Return a JSON error response."""
    return jsonify({"error": str(exc)}), status


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/move_to", methods=["POST"])
def move_to():
    """Move end-effector to an absolute [x, y, z] position."""
    try:
        body = request.get_json(force=True) or {}
        x = float(body["x"])
        y = float(body["y"])
        z = float(body["z"])
        result = robot.move_to(x, y, z)
        return _ok(result)
    except Exception as e:
        return _err(e)


@app.route("/pick", methods=["POST"])
def pick():
    """Pick up the cube using the side-approach strategy."""
    try:
        result = robot.pick()
        return _ok(result)
    except Exception as e:
        return _err(e)


@app.route("/place", methods=["POST"])
def place():
    """Place the held object at an absolute [x, y, z] position."""
    try:
        body = request.get_json(force=True) or {}
        x = float(body["x"])
        y = float(body["y"])
        z = float(body["z"])
        result = robot.place(x, y, z)
        return _ok(result)
    except Exception as e:
        return _err(e)


@app.route("/gripper", methods=["POST"])
def gripper():
    """Set gripper openness: 0.0 = fully open, 1.0 = fully closed."""
    try:
        body = request.get_json(force=True) or {}
        value = float(body["value"])
        result = robot.gripper(value)
        return _ok(result)
    except Exception as e:
        return _err(e)


@app.route("/state", methods=["GET"])
def state():
    """Return a full telemetry snapshot of the arm and simulation."""
    try:
        return _ok(robot.get_state())
    except Exception as e:
        return _err(e)


@app.route("/camera", methods=["GET"])
def camera():
    """Capture an image from the wrist camera and return the file path."""
    try:
        # Write to a temp file that persists after the call so the caller
        # can read it.  Using delete=False and a .png suffix.
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", delete=False, prefix="wrist_cam_"
        )
        tmp.close()  # close so save_camera_image can write to it on Windows

        saved_path = robot.save_camera_image(filepath=tmp.name)
        return _ok({"path": os.path.abspath(saved_path)})
    except Exception as e:
        return _err(e)


@app.route("/reset_home", methods=["POST"])
def reset_home():
    """Return the arm to its home rest position."""
    try:
        robot.reset_home()
        # reset_home() returns None, so fall back to current state.
        return _ok(robot.get_state())
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Additional endpoints
# ---------------------------------------------------------------------------

import mujoco as _mujoco  # already a transitive import via robot_api; alias avoids shadowing


@app.route("/collisions", methods=["GET"])
def collisions():
    """Return the current contact pairs MuJoCo has detected this step.

    Reads robot.data.contact[:robot.data.ncon] — no extra physics stepping.
    Geom IDs are resolved to human-readable names via mj_id2name.
    """
    try:
        ncon = robot.data.ncon
        contacts = []
        for i in range(ncon):
            c = robot.data.contact[i]
            name1 = _mujoco.mj_id2name(
                robot.model, _mujoco.mjtObj.mjOBJ_GEOM, c.geom1
            ) or f"geom_{c.geom1}"
            name2 = _mujoco.mj_id2name(
                robot.model, _mujoco.mjtObj.mjOBJ_GEOM, c.geom2
            ) or f"geom_{c.geom2}"
            contacts.append({"geom1": name1, "geom2": name2})
        return _ok({"in_contact": ncon > 0, "contacts": contacts})
    except Exception as e:
        return _err(e)


@app.route("/scenario", methods=["POST"])
def scenario():
    """Run a named pick-and-place scenario.

    Body: {"name": "A" | "B" | "C"}

    A  — Pick cube, place at right pad (0.15, -0.18, 0.015), return home.
    B  — Pick cube, place at left pad (-0.15, -0.18, 0.015), return home.
    C  — Inspection wave: move up, save photo, sweep left/right, return home.

    Returns get_state() on success, or {"error": ...} with HTTP 400.
    """
    try:
        body = request.get_json(force=True) or {}
        name = str(body.get("name", "")).upper()

        if name == "A":
            robot.pick()
            robot.place(0.15, -0.18, 0.015)
            robot.reset_home()
        elif name == "B":
            robot.pick()
            robot.place(-0.15, -0.18, 0.015)
            robot.reset_home()
        elif name == "C":
            robot.move_to(0.0, -0.20, 0.16)
            robot.save_camera_image("inspection_high.png")
            robot.move_to(-0.12, -0.22, 0.12)
            robot.move_to(0.12, -0.22, 0.12)
            robot.reset_home()
        else:
            return _err("unknown scenario")

        return _ok(robot.get_state())
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[server] Starting Flask REST server on http://0.0.0.0:8765")
    print("[server] Endpoints: /move_to  /pick  /place  /gripper  /state  /camera  /reset_home")
    app.run(
        host="0.0.0.0",
        port=8765,
        debug=False,
        use_reloader=False,   # Reloader would spawn a second RobotAPI instance
        threaded=False,       # Single-threaded: prevents concurrent requests racing on shared robot state
    )
