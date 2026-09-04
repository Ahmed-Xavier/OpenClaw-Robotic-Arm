# OpenClaw Embodied Robotics (MuJoCo Edition)

> Giving OpenClaw a virtual robotic body — running entirely on MuJoCo, no ROS2/Gazebo required.

This project integrates **OpenClaw** with a simulated robotic arm using **MuJoCo + Gymnasium**
instead of the original ROS2/Gazebo/MoveIt2 plan (see the `legacy-ros2-gazebo-plan` branch for
that earlier direction — kept for reference, not actively developed).

The goal is a locally-running embodied AI agent that can perceive a simulated environment,
reason about tasks, control a robotic arm, and interact with objects — developed and testable
entirely on a normal Windows machine, no Linux/ROS2 install required for the simulation layer.

---

## Why the pivot from the original plan

The original architecture (see the legacy branch) targeted ROS2 + Gazebo + MoveIt2, which
requires a full Ubuntu/ROS2 install. This version reaches the same capabilities using:

| Original plan | This version |
|---|---|
| Gazebo simulation | MuJoCo (`mujoco` + `dm_control`) |
| MoveIt2 (inverse kinematics, planning) | Hand-built Operational Space Controller (OSC) |
| ROS2 topics for robot state | Direct `physics.data` access via `dm_control` |
| Ubuntu 24.04 required | Runs on Windows (Python + pip only) |

---

## Project Goal

- Move the arm to precise positions (**done** — OSC-controlled, live-tracked)
- Pick and place objects (**done** — see `robot_api.py`)
- Add camera-based perception (**in progress**)
- Control via natural language through OpenClaw (**next**)
- Play Tic-Tac-Toe / Chess in simulation (**future**)

---

## Current Architecture

```
                 USER
                   │
                   ▼
            ┌─────────────┐
            │   OpenClaw  │   (not yet wired up — next step)
            └──────┬──────┘
                   │
            OpenClaw Skill (CLI subprocess calls, TBD)
                   │
                   ▼
            ┌─────────────┐
            │ robot_api.py│
            │             │
            │ move_to()   │
            │ pick()      │
            │ place()     │
            │ gripper()   │
            │ get_state() │
            └──────┬──────┘
                   │
                   ▼
        MuJoCo + Gymnasium env
     (manipulator_mujoco/AuboI5Env-v0)
                   │
                   ▼
          OSC Controller (IK)
                   │
                   ▼
             MuJoCo physics
```

---

## Repository Structure

```
openclaw-robotic-arm/
│
├── README.md
├── robot_api.py          # Robot abstraction layer (move_to, pick, place, gripper)
├── requirements.txt
│
└── (planned)
    ├── openclaw_skill/    # OpenClaw-callable CLI wrapper + skill.md
    ├── perception/        # Camera-based object detection (MuJoCo render + CV)
    └── tasks/             # Tic-tac-toe, chess logic
```

---

## Setup

This depends on the [Manipulator-Mujoco](https://github.com/ian-chuang/Manipulator-Mujoco)
environment package. Known-working versions (pinned due to dm_control/MuJoCo internal API
churn between versions):

```
pip install "mujoco==3.2.1" "dm-control==1.0.22"
```

Then clone and install Manipulator-Mujoco per its own README, and this repo's `robot_api.py`
imports its registered Gymnasium environment directly.

---

## Robot Abstraction API

`robot_api.py` exposes a hardware-independent interface, designed so the same calls will
later work against a real arm without changing the calling code:

```python
robot.move_to(x, y, z)
robot.pick()
robot.place(x, y, z)
robot.gripper(value)      # 0.0 = open, 1.0 = closed
robot.get_state()         # {"holding_cube": bool, "eef_position": [...], "cube_position": [...]}
```

Key design decisions (learned the hard way, see commit history / dev notes):
- The cube's position is **read live from physics on every `pick()` call**, never cached —
  an earlier version silently fell back to a hardcoded guess when name-detection failed.
- The end-effector site is found by **substring match**, not exact name, since `dm_control`
  prefixes site names (e.g. `aubo_i5/eef_site`) unpredictably depending on model assembly.
- The fingertip-to-eef vertical offset is **measured from the model at startup**, with a loud
  warning if it has to fall back to a guessed constant, rather than silently trusting a number
  that might not match the actual gripper geometry.

---

## Next Steps

1. Wrap `robot_api.py` in a CLI script (`arm_api.py`) callable by OpenClaw as a subprocess,
   following the pattern used in
   [AI-Robotic-Arm-RDK-S100-OpenClaw](https://github.com/proknowdiy/AI-Robotic-Arm-RDK-S100-OpenClaw).
2. Write the OpenClaw skill markdown describing when/how to call each command.
3. Add camera-based perception (`physics.render()` + object detection) so the agent can
   locate objects itself instead of relying on a single hardcoded cube.
4. Extend to Tic-Tac-Toe and Chess once perception + manipulation are both reliable.

---

## Legacy Plan

The original ROS2 + Gazebo + MoveIt2 architecture is preserved on the
[`legacy-ros2-gazebo-plan`](../../tree/legacy-ros2-gazebo-plan) branch for reference and
future revisiting if/when a full Ubuntu/ROS2 environment is available.
