# OpenClaw Embodied Robotics

> Giving OpenClaw a virtual robotic body — starting entirely in simulation.

This project explores integrating **OpenClaw** with a simulated robotic arm using **ROS 2, Gazebo, MoveIt 2, and RGB-D perception**.

The goal is to create a locally-running embodied AI agent that can perceive a simulated environment, reason about tasks, control a robotic arm, and interact with objects.

The project starts entirely in simulation so the complete system can be developed and tested without physical robotics hardware.

---

## 🎯 Project Goal

The long-term goal is to turn OpenClaw from a software-only agent into an **embodied agent** capable of interacting with the physical world.

Initial target capabilities:

- Pick and place objects
- Sort objects
- Move objects based on natural-language instructions
- Play Tic-Tac-Toe
- Play chess
- Interact conversationally with the user
- Understand its environment through vision
- Verify whether actions succeeded

Eventually, the same architecture should be transferable to an affordable real robotic arm.

---

## 🧠 Overall Architecture

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │   OpenClaw  │
                    │             │
                    │ Local LLM   │
                    │ Memory      │
                    │ Planning    │
                    │ Tools       │
                    └──────┬──────┘
                           │
                    OpenClaw Skills
                           │
                           ▼
                    ┌─────────────┐
                    │ Robot API   │
                    │             │
                    │ observe()   │
                    │ pick()      │
                    │ place()     │
                    │ move_to()   │
                    │ gripper()   │
                    └──────┬──────┘
                           │
                           ▼
                        ROS 2
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        ┌───────────┐             ┌────────────┐
        │ MoveIt 2  │             │ Perception │
        │           │             │            │
        │ Planning  │             │ RGB-D      │
        │ IK        │             │ Detection  │
        │ Collision │             │ 3D Poses   │
        └─────┬─────┘             └──────┬─────┘
              │                          │
              └──────────┬───────────────┘
                         ▼
                    ┌──────────┐
                    │ Gazebo   │
                    │          │
                    │ UR5e     │
                    │ Objects  │
                    │ Camera   │
                    └──────────┘
```

---

## 📦 Source Projects

This repository combines ideas and components from two existing projects.

### 1. Universal Robots ROS 2 Gazebo Simulation

Used as the foundation for the simulated robotic body.

Repository:

https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation

Provides:

- Universal Robot models
- UR5e simulation
- Gazebo integration
- ROS 2 integration
- `ros2_control`
- MoveIt 2 integration
- Robot descriptions
- Simulation launch files

The UR5e is currently the primary simulated robot.

### 2. AI Robotic Arm RDK-S100 OpenClaw

Repository:

https://github.com/proknowdiy/AI-Robotic-Arm-RDK-S100-OpenClaw

This project is used primarily as an **architectural reference** for connecting OpenClaw to a robotic arm.

Important concepts to adapt:

- OpenClaw robot skills
- Robot API abstraction
- High-level commands
- Scene observation
- Pick/place actions
- Separation between agent logic and hardware control

The hardware-specific portions will not be used directly because this project targets Gazebo/ROS 2 simulation first.

---

# 🔀 Merge / Integration Plan

The two projects serve different purposes.

We are **not simply combining their files together**.

Instead:

```text
Universal Robots Repository
            │
            ▼
     Simulation Layer
            │
     UR5e + Gazebo
     ROS 2
     MoveIt 2
     ros2_control
            │
            ▼
       Robot Interface
            ▲
            │
RDK-S100 OpenClaw Project
            │
            ▼
      OpenClaw Layer
```

The RDK-S100 project provides the OpenClaw integration pattern.

The Universal Robots project provides the simulated robot.

---

# 🏗️ Target Repository Structure

The final project is expected to evolve toward:

```text
openclaw-embodied-robotics/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── openclaw/
│   ├── skills/
│   │   └── robotics/
│   └── configuration/
│
├── robot_api/
│   ├── robot.py
│   ├── perception.py
│   ├── manipulation.py
│   └── gripper.py
│
├── ros2_ws/
│   └── src/
│       ├── robot_bridge/
│       ├── robot_perception/
│       └── robot_tasks/
│
├── simulation/
│   ├── worlds/
│   ├── models/
│   ├── cameras/
│   └── objects/
│
├── perception/
│   ├── object_detection/
│   ├── depth/
│   └── pose_estimation/
│
├── tasks/
│   ├── pick_and_place/
│   ├── sorting/
│   ├── tic_tac_toe/
│   └── chess/
│
└── docs/
    ├── architecture.md
    ├── setup.md
    ├── perception.md
    └── openclaw-integration.md
```

The exact structure may change during development.

---

# 🤖 Robot Abstraction

OpenClaw should not be coupled to the UR5e.

It should interact with a generic robot API:

```python
observe_scene()

find_object("red_cube")

move_to(x, y, z)

pick("red_cube")

place("red_cube", x, y, z)

open_gripper()

close_gripper()

get_robot_state()
```

The implementation can change.

### Simulation

```text
OpenClaw
   ↓
Robot API
   ↓
ROS 2
   ↓
MoveIt 2
   ↓
Gazebo UR5e
```

### Future real robot

```text
OpenClaw
   ↓
Robot API
   ↓
ROS 2
   ↓
Real Robot Driver
   ↓
Physical Arm
```

This abstraction is intentional.

---

# 👁️ Perception

The simulation will use an RGB-D camera.

The camera provides:

```text
RGB Image
+
Depth Image
```

These can be used to determine the 3D location of objects.

Example:

```json
{
  "object": "red_cube",
  "position": {
    "x": 0.42,
    "y": -0.17,
    "z": 0.04
  },
  "confidence": 0.97
}
```

OpenClaw does not need to calculate the object's distance itself.

```text
Camera
   ↓
Vision
   ↓
Object Detection
   ↓
Depth
   ↓
3D Object Pose
   ↓
OpenClaw
```

---

# 🧠 Division of Responsibilities

## OpenClaw

Responsible for:

- Natural-language interaction
- Task planning
- Reasoning
- Memory
- Tool selection
- High-level decision making

Example:

> "Pick up the red cube and put it in the box."

## Perception

Responsible for:

- Detecting objects
- Estimating object positions
- Processing RGB images
- Processing depth
- Producing structured scene information

Example:

```text
red_cube
position = [0.42, -0.17, 0.04]
```

## MoveIt 2

Responsible for:

- Inverse kinematics
- Motion planning
- Collision checking
- Trajectory generation

OpenClaw should not directly calculate joint trajectories.

## Gazebo

Responsible for:

- Simulated robot
- Simulated environment
- Physics
- Sensors
- Objects
- Cameras
- Robot interaction

---

# 🔄 Embodied AI Loop

```text
        ┌───────────────┐
        │    Observe    │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │    Perceive   │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │    OpenClaw   │
        │     Reasons   │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │     Act       │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Verify Result │
        └───────┬───────┘
                │
                └───────────────► Observe
```

The agent should be able to verify whether an action actually succeeded.

---

# 🚀 Development Roadmap

## Phase 0 — Environment

- [ ] Install Ubuntu 24.04
- [ ] Install ROS 2 Jazzy
- [ ] Install Gazebo
- [ ] Install MoveIt 2
- [ ] Build the Universal Robots simulation
- [ ] Launch the UR5e

## Phase 1 — Robot Simulation

- [ ] Spawn UR5e
- [ ] Verify joint controllers
- [ ] Control robot manually
- [ ] Verify MoveIt 2
- [ ] Test inverse kinematics
- [ ] Test collision detection

## Phase 2 — Simulation Environment

- [ ] Create table
- [ ] Add cubes
- [ ] Add containers
- [ ] Add simple objects
- [ ] Create reusable simulation worlds

## Phase 3 — Vision

- [ ] Add RGB camera
- [ ] Add depth camera
- [ ] View camera feed in RViz/Gazebo
- [ ] Subscribe to ROS image topics
- [ ] Detect objects
- [ ] Convert image coordinates to 3D positions
- [ ] Publish object poses

## Phase 4 — Robot API

Create a hardware-independent interface:

```python
observe_scene()
find_object(name)
move_to_pose(pose)
pick(name)
place(name, pose)
open_gripper()
close_gripper()
get_robot_state()
```

## Phase 5 — OpenClaw Integration

- [ ] Create OpenClaw robotics skill
- [ ] Connect skill to Robot API
- [ ] Expose perception tools
- [ ] Expose manipulation tools
- [ ] Test natural-language commands
- [ ] Connect local LLM

## Phase 6 — First Embodied Task

Target:

> "Pick up the red cube."

Expected behavior:

```text
User
 ↓
OpenClaw
 ↓
find_object("red_cube")
 ↓
3D pose
 ↓
MoveIt 2
 ↓
Gazebo UR5e
 ↓
Grasp
 ↓
Verify
 ↓
OpenClaw
```

## Phase 7 — Manipulation Tasks

- [ ] Pick and place
- [ ] Object sorting
- [ ] Move objects between containers
- [ ] Multi-object tasks
- [ ] Failure detection
- [ ] Recovery behaviors

## Phase 8 — Games

### Tic-Tac-Toe

The agent should:

1. Observe the board.
2. Detect the player's move.
3. Determine the game state.
4. Choose its move.
5. Move the appropriate piece.
6. Verify the placement.
7. Continue until the game ends.

### Chess

```text
Camera
   ↓
Board Detection
   ↓
Piece Recognition
   ↓
Board State
   ↓
Chess Engine
   ↓
OpenClaw
   ↓
Move Planning
   ↓
Robot Arm
```

A dedicated chess engine can handle chess calculation while OpenClaw handles interaction, planning, and physical execution.

---

# 🧪 Simulation First

This project intentionally starts without physical hardware.

Advantages:

- No hardware cost
- No servo damage
- No calibration hardware
- Easy experimentation
- Repeatable tests
- Faster development
- Easy debugging
- Ability to reset the environment instantly

The simulation is the primary development environment.

---

# 🔌 Future Hardware

Once the simulated system is reliable, the same high-level architecture should be connectable to an affordable physical robotic arm.

```text
                    SAME
              OpenClaw Robot API
                     │
          ┌──────────┴──────────┐
          │                     │
       Simulation             Hardware
          │                     │
       Gazebo              Robot Driver
          │                     │
        UR5e                Physical Arm
```

The goal is to minimize changes to OpenClaw when switching from simulation to hardware.

---

# 🛠️ Planned Technology Stack

| Component | Technology |
|---|---|
| Agent | OpenClaw |
| LLM | Local model |
| Robotics Middleware | ROS 2 |
| Simulation | Gazebo |
| Motion Planning | MoveIt 2 |
| Control | ros2_control |
| Robot | Universal Robots UR5e |
| Vision | RGB-D camera |
| Object Detection | TBD |
| Depth | Simulated depth sensor |
| Chess Engine | TBD |
| Operating System | Ubuntu 24.04 |

---

# 📚 Project Status

**Current stage:** Architecture / initial setup

Current objectives:

1. Set up the ROS 2 + Gazebo robotic arm simulation.
2. Understand the OpenClaw robotic-arm architecture.
3. Design the generic robot API.
4. Add RGB-D perception.
5. Connect OpenClaw to the simulated robot.

---

# ⚠️ Licensing

This project uses existing open-source robotics projects as dependencies and architectural references.

The original projects retain their respective licenses and copyrights.

Before redistributing copied source code from upstream repositories, verify and comply with their individual licenses.

---

# 🌟 Vision

> **Give OpenClaw a body.**

Not just a chatbot that can describe actions, but an agent that can:

```text
SEE 👁️
  ↓
UNDERSTAND 🧠
  ↓
PLAN 🗺️
  ↓
ACT 🤖
  ↓
VERIFY 🔎
  ↓
REMEMBER 🧠
```

Starting with a simulated robotic arm, this project explores how a local AI agent can interact with the physical world through perception, reasoning, and robotic manipulation.

---

## Future Goal

```text
                    OpenClaw
                       │
              Local Intelligence
                       │
             ┌─────────┴─────────┐
             │                   │
          Virtual              Real
           Body                Body
             │                   │
          Gazebo             Robotic Arm
             │                   │
             └─────────┬─────────┘
                       │
                 Embodied AI
```

**The simulation is the first body.**
