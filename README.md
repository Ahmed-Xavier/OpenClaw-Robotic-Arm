# OpenClaw Embodied Robotics — SO-100 MuJoCo Simulation

> A locally-running robotic arm simulation using the **Hugging Face LeRobot SO-ARM100 (SO-100)**
> and **MuJoCo** — no ROS2, no Gazebo, no Ubuntu required.

---

## Overview

This project simulates the **SO-100** 5-DOF robotic arm in MuJoCo with a full pick-and-place
pipeline, wrist camera, and an interactive terminal control menu.

The simulation is hardware-independent: the same `robot_api.py` interface is designed to work
identically when wired to the real SO-100 arm via the LeRobot hardware layer.

---

## Why MuJoCo + SO-100

The original plan (see `legacy-ros2-gazebo-plan` branch) used ROS2 + Gazebo + MoveIt2 with an
Aubo i5 industrial arm — which requires a full Ubuntu/ROS2 install and is overkill for prototyping.

| Old plan | This version |
|---|---|
| Gazebo simulation | MuJoCo (native Python, no ROS) |
| MoveIt2 IK planner | Custom Jacobian pseudo-inverse IK |
| Aubo i5 industrial arm (6-DOF, $15k+) | SO-100 maker arm (5-DOF, ~$100) |
| Ubuntu 24.04 required | Runs on Windows (Python + pip only) |
| ROS2 topics for state | Direct `mujoco.MjData` access |

---

## Robot Platform

**SO-ARM100 (SO-100)** by Hugging Face LeRobot — a low-cost, open-source 5-DOF desktop
robotic arm with a parallel jaw gripper.

- **Joints:** Rotation (base) → Pitch (shoulder) → Elbow → Wrist Pitch → Wrist Roll → Jaw
- **Gripper:** Fixed Jaw + Moving Jaw (position-controlled, 0 = closed, 1.5 rad = open)
- **Wrist camera:** `wrist_camera` site mounted on the Fixed Jaw, looking at the grasp site
- **MuJoCo model:** vendored under `third_party/so_arm100/` (MuJoCo Menagerie assets)

---

## Architecture

```
                 USER
                   │
         terminal input (1–10)
                   │
                   ▼
            ┌─────────────┐
            │  robot_api  │   Interactive menu + scenario runner
            │             │
            │ move_to()   │   IK → smooth joint interpolation
            │ pick()      │   Side-approach grab (no top-down collision)
            │ place()     │   Hover → lower → release → retract
            │ gripper()   │   0.0 open … 1.0 closed
            │ get_state() │   Live telemetry snapshot
            │ capture_image() │  RGB from wrist camera
            └──────┬──────┘
                   │
                   ▼
         MuJoCo physics engine
         (so_arm100 MJCF model)
                   │
            Jacobian IK solver
            (pseudo-inverse, 5-DOF)
```

---

## Repository Structure

```
OpenClaw-Robotic-Arm/
│
├── README.md
├── robot_api.py              # Robot abstraction layer + interactive menu
├── requirements.txt
│
└── third_party/
    └── so_arm100/            # SO-100 MuJoCo Menagerie model (vendored)
        ├── interactive_scene.xml   # Full scene: arm + table + red cube + target pads
        ├── so_arm100.xml           # Arm + gripper MJCF definition
        └── assets/                 # STL meshes
```

---

## Installation

This section covers setting up the MuJoCo simulation and Flask REST server on a local machine.

### Prerequisites

- **Operating System:** Windows 10/11 (tested natively on Windows without requiring WSL or Ubuntu).
- **Python:** Python 3.10+ (tested on Python 3.11.9).
- **Git**

> [!NOTE]
> All MuJoCo 3D models, scene definitions (`interactive_scene.xml`, `so_arm100.xml`), and STL mesh assets are vendored directly in `third_party/so_arm100/`. No additional asset downloads, git submodule initializations, or system environment variables are required.

### 1. Clone & Environment Setup

Clone the repository and create a Python virtual environment:

```powershell
# Clone the repository
git clone https://github.com/Ahmed-Xavier/OpenClaw-Robotic-Arm.git
cd OpenClaw-Robotic-Arm

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

The repository includes a `requirements.txt` specifying the Flask server dependency. The simulation physics and camera pipeline additionally require `mujoco`, `numpy`, and `pillow`:

```powershell
pip install --upgrade pip
pip install -r requirements.txt mujoco numpy pillow
```

> **Core Dependencies:**
> - `flask` (HTTP REST API layer)
> - `mujoco` (>= 3.2.0, physics simulation and passive viewer)
> - `numpy` (>= 1.24, kinematics and state math)
> - `pillow` (offscreen camera image rendering and saving)

### 3. Running the Server

Start the simulation and HTTP REST server:

```powershell
python server.py
```

*(Alternatively, run `Run Server.bat` on Windows).*

> [!WARNING]
> **Single-Instance Only:** `server.py` instantiates a single global `RobotAPI(render=True)` instance holding the MuJoCo simulation and viewer handle. It runs with `use_reloader=False` and `threaded=False`. Running multiple instances or enabling the Werkzeug reloader will spawn duplicate processes that fight over the MuJoCo viewer and physics state. Ensure only **one** instance of `server.py` is running.

### 4. What a Successful Run Looks Like

When launched successfully:

1. **MuJoCo Viewer Opens:** A native graphical window opens automatically displaying the SO-100 5-DOF arm in its rest pose over the workspace table, along with the red target cube and placement pads. The viewer is interactive and maintains real-time physics (supporting perturbations via `Ctrl` + Right-Click mouse dragging).
2. **Flask Server Binds:** The server listens on `http://0.0.0.0:8765` (accessible locally at `http://127.0.0.1:8765` or `http://localhost:8765`).
3. **Console Output:**
   ```text
   [robot_api] SO-100 arm initialized and ready.
   [server] Starting Flask REST server on http://0.0.0.0:8765
   [server] Endpoints: /move_to  /pick  /place  /gripper  /state  /camera  /reset_home
    * Serving Flask app 'server'
    * Debug mode: off
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    * Running on all addresses (0.0.0.0)
    * Running on http://127.0.0.1:8765
    * Running on http://<local-ip>:8765
   Press CTRL+C to quit
   ```

### 5. Available REST Endpoints

External clients, scripts, or agent drivers interact with the simulation using standard HTTP JSON calls on port `8765`:

| Endpoint | Method | Payload / Parameters | Description |
|---|---|---|---|
| `/move_to` | `POST` | `{"x": float, "y": float, "z": float}` | Moves the end-effector to target coordinates via Jacobian IK. |
| `/pick` | `POST` | `{}` | Executes the horizontal side-approach pickup sequence for the red cube. |
| `/place` | `POST` | `{"x": float, "y": float, "z": float}` | Moves the held object to target coordinates, lowers, and releases. |
| `/gripper` | `POST` | `{"value": float}` | Controls gripper openness (`0.0` = closed, `1.0` = open). |
| `/state` | `GET` | _None_ | Returns full telemetry snapshot (EEF, cube, gripper, joint angles in radians, sim time). |
| `/camera` | `GET` | _None_ | Renders wrist camera view and returns `{"path": "<absolute path to PNG>"}`. |
| `/reset_home` | `POST` | `{}` | Returns the robotic arm to the home rest position. |
| `/collisions` | `GET` | _None_ | Returns contact pairs detected by MuJoCo during the current step. |
| `/scenario` | `POST` | `{"name": "A" \| "B" \| "C"}` | Executes predefined routines (Scenario A: Right pad, B: Left pad, C: Inspection wave). |

### Troubleshooting

- **Arm does not move in viewer, but API calls return HTTP 200:**
  This typically occurs if a zombie or duplicate `server.py` process is already running in the background holding port `8765`. Any new request hits the old process (or an unrendered instance), leaving the visible viewer untouched.
  
  To check for and terminate duplicate processes on Windows:
  ```powershell
  # Find process IDs listening on port 8765
  netstat -ano | findstr :8765

  # Kill the offending process by its PID
  taskkill /PID <PID> /F
  ```

---

## Setup

```bash
pip install mujoco pillow numpy
```

Run the simulation:
```bash
python robot_api.py
```

The MuJoCo viewer window opens automatically. Use the terminal menu to control the arm.

---

## Interactive Control Menu

```
========================================
     SO-100 ROBOTIC ARM CONTROL MENU
========================================
  [1]  Info      — Live telemetry (arm, cube, gripper, joints)
  [2]  Grab      — Pick up the red cube (live tracked)
  [3]  Place     — Drop cube at Right or Left target pad
  [4]  Manual    — Move arm to custom X Y Z coordinates
  [5]  Gripper   — Open / Close gripper manually
  [6]  Camera    — Save wrist camera snapshot to disk
  --------------------------------------
  SCENARIOS:
  [7]  Scenario A — Pick & Place to Right target, return home
  [8]  Scenario B — Pick & Place to Left target, return home
  [9]  Scenario C — Inspection wave + high-angle camera photo
  [10] Reset      — Return arm to home pose
  [q]  Quit
========================================
```

---

## Robot API

`robot_api.py` exposes a clean, hardware-independent interface:

```python
from robot_api import RobotAPI

robot = RobotAPI(render=True)

robot.move_to(x, y, z)             # Move end-effector to world position
robot.pick()                        # Grab the red cube (side-approach)
robot.place(x, y, z)               # Place held object at target
robot.gripper(value)                # 0.0 = fully open, 1.0 = fully closed
robot.get_state()                   # Returns live telemetry dict
robot.capture_image()               # Returns RGB array from wrist camera
robot.save_camera_image("out.png")  # Saves wrist camera snapshot to disk
robot.reset_home()                  # Return arm to home pose
robot.close()                       # Shut down viewer and simulation
```

### Telemetry (`get_state()`)

```python
{
    "holding_cube": bool,
    "dist_to_cube": float,          # metres, EEF to cube centre
    "eef_position":  {"x", "y", "z"},
    "cube_position": {"x", "y", "z"},
    "gripper": {
        "openness": float,          # 0.0 closed … 1.0 open
        "raw_rad":  float,          # jaw joint angle in radians
        "is_closed": bool
    },
    "joints_rad": {                 # all 6 joint angles
        "Rotation (Base)": float,
        "Pitch (Shoulder)": float,
        "Elbow": float,
        "Wrist Pitch": float,
        "Wrist Roll": float,
        "Jaw (Gripper)": float
    },
    "sim_time": float               # seconds elapsed in simulation
}
```

---

## Pick Logic — Side Approach

The `pick()` method uses a **horizontal side-approach** instead of a top-down hover+descend,
which caused the fixed jaw tip to collide with the top of the object.

```
1. Open gripper
2. Compute approach direction: arm-base → cube (XY plane)
3. Move to standoff position (12 cm behind cube, at cube height)  ← no hover
4. Slide in horizontally to the cube centre
5. Settle (120 physics steps) — arm damps oscillation
6. Close gripper (80 steps)
7. Hold grip (0.5 s dwell) — grip fully established before lifting
8. Lift straight up
```

Guard: if `_holding` is already `True`, `pick()` skips immediately and returns current state.

---

## IK Solver

5-DOF Jacobian pseudo-inverse IK with damped least squares:

- Seeds from current joint configuration for smooth, short-path motion
- Per-joint limit clamping after every step
- Damping factor scales with distance to target for stability near the goal
- Runs in a shadow `MjData` copy — does not disturb live physics until `_step_to_ctrl()` is called

---

## Next Steps

- [ ] Wire `robot_api.py` to the **real SO-100** via LeRobot hardware driver
- [ ] Add natural language control through **OpenClaw** (CLI subprocess skill)
- [ ] Camera-based object detection — locate objects by vision, not by physics state
- [ ] Extend to Tic-Tac-Toe / Chess manipulation tasks

---

## Considering: MoveIt Integration

The current custom Jacobian pseudo-inverse IK is sufficient for a single 5-DOF arm in a known,
static workspace — but it has no notion of collision-awareness or path planning around
obstacles, only point-to-point motion. MoveIt would be worth revisiting if this project grows
in any of these directions:

- **Collision-aware planning** — MoveIt builds a 3D planning scene (from the URDF/MJCF + sensor
  data) and routes around obstacles instead of assuming a clear straight-line path
- **Multi-arm coordination** — if a second arm is added to the scene, MoveIt's planning scene can
  coordinate both arms to avoid collisions with each other, which the current per-arm `move_to()`
  cannot do
- **Grasp planning** — MoveIt has built-in grasp generation and pick-place pipelines that plug
  into perception, rather than hand-coded fixed approach vectors like the current side-approach
  `pick()` logic
- **ROS2 ecosystem fit** — if this ever moves onto a ROS2-based platform (see legacy plan below),
  MoveIt is the standard manipulation stack most labs already expect to plug into

Tradeoff: MoveIt brings real setup overhead (SRDF config, planning scene, controller
integration) that isn't worth it for the current single-arm, static, MuJoCo-only setup — the
custom IK stays lighter and faster to iterate on until one of the above becomes an actual need.

---

## Legacy Plan

The original ROS2 + Gazebo + MoveIt2 architecture is preserved on the
[`legacy-ros2-gazebo-plan`](../../tree/legacy-ros2-gazebo-plan) branch for reference.
