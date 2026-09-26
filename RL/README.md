# Phase 1: UR5e End-Effector Reaching via Reinforcement Learning

This directory contains **Phase 1** of the reinforcement learning system for robotic manipulation in the OpenClaw repository.

---

## 1. What Phase 1 Is

Phase 1 establishes the first minimal, working, continuous-control reinforcement learning pipeline. 

The primary goal is simple and isolated:
> **Train a PPO policy whose only objective is to move the UR5e end-effector (EEF) to a randomly sampled reachable 3D target position.**

This forms the lowest learned layer of the robot control stack.

### Architecture Principle

```text
High-Level Planner / LLM (Future)
        ↓
High-Level Skill Layer (e.g., pick, place, push)
        ↓
RL Policy (Low-level learned skill: ReachEnv)
        ↓
Operational Space Controller (OSC)
        ↓
MuJoCo Physics Simulation / Physical UR5e
```

**Key Isolation Guarantees**:
- Completely independent of LLMs, prompts, and tool calling.
- Does not modify or depend on [`robot_api.py`](../robot_api.py).
- Completely isolated under `RL/`.

---

## 2. Environment Design (`ReachEnv`)

- **Simulation Engine**: `robosuite 1.5.2` (`Lift` environment with `UR5e` robot).
- **Control Mode**: Operational Space Control (`OSC_POSE`) at `control_freq=20` Hz (50 ms per step).
- **Episode Horizon**: 100 steps (5.0 seconds simulation time).

### Observation Space
A compact, fully Markovian, 9-dimensional numerical vector (`dtype=float32`):
| Indices | Dimension | Description | Finite Bounds |
| :--- | :--- | :--- | :--- |
| `0:3` | 3 | Current EEF position $(x, y, z)$ | $[-5.0, 5.0]$ |
| `3:6` | 3 | Sampled Target position $(x_t, y_t, z_t)$ | $[-5.0, 5.0]$ |
| `6:9` | 3 | Relative displacement vector $(x_t - x, y_t - y, z_t - z)$ | $[-5.0, 5.0]$ |

*(Note: Raw camera images, cube object poses, and high-dimensional proprioception arrays are deliberately filtered out).*

### Action Space
- **Policy Action Space**: Continuous 3D Cartesian delta translations:
  $$\mathbf{a}_t = [dx, dy, dz] \in [-1.0, 1.0]^3$$
- **Action Mapping**: The wrapper constructs the full 7-dimensional robosuite OSC action:
  ```python
  full_action = [dx, dy, dz, 0.0, 0.0, 0.0, -1.0]
  #                  ^           ^              ^
  #             Translation   Neutral Ori    Open Gripper
  ```

### Target Sampling
At every episode reset:
1. The simulation resets to the standard UR5e configuration.
2. The initial EEF position is recorded ($p_0 \approx [-0.23, -0.01, 0.97]$ m).
3. A reachable target is sampled nearby with $\Delta \in [-0.12, 0.12]$ m.
4. The target is clamped to a safe conservative workspace above the table:
   - $X \in [-0.35, 0.05]$ m
   - $Y \in [-0.20, 0.20]$ m
   - $Z \in [0.86, 1.10]$ m (table plane is at $Z = 0.80$ m)

---

## 3. Reward Function

A dense, progress-driven reward structure:
$$r_t = 10 \cdot (d_{t-1} - d_t) - 0.1 \cdot d_t - 0.01 \|\mathbf{a}_t\|^2 + r_{\text{bonus}}$$

where:
- $d_t = \|\mathbf{p}_{\text{eef}} - \mathbf{p}_{\text{target}}\|$ is the Euclidean distance to the target.
- $(d_{t-1} - d_t)$ provides immediate positive reward for reducing distance.
- $-0.1 \cdot d_t$ penalizes remaining far from the target.
- $-0.01 \|\mathbf{a}_t\|^2$ penalizes jerky, excessive actions.
- $r_{\text{bonus}} = +5.0$ awarded upon reaching success.

### Success Condition
An episode is classified as a **SUCCESS** and terminates immediately when:
$$\|\mathbf{p}_{\text{eef}} - \mathbf{p}_{\text{target}}\| < 0.03 \text{ m} \quad (3 \text{ cm})$$

---

## 4. What We Are Deliberately NOT Doing Yet

To keep Phase 1 small, verifiable, and fast:
- ❌ **No Vision / Cameras**: No RGB/depth renders in the observation space.
- ❌ **No Grasping**: Gripper remains fixed open (`-1.0`).
- ❌ **No Cubes / Object Manipulation**: The cube from robosuite's `Lift` is ignored.
- ❌ **No LLM / Agent Coupling**: No XavierClaw/Ollama integration.
- ❌ **No Sim2Real**: Pure MuJoCo continuous control benchmark.

---

## 5. Usage Commands

### Sanity Check
Test environment initialization, observation/action shapes, and stepping:
```bash
python RL/envs/reach_env.py
```

### Training
Train the PPO agent on the UR5e reaching task (saves weights to `RL/models/ur5e_reach_ppo.zip`):
```bash
# Standard training (50,000 timesteps)
python RL/train.py

# Custom timesteps or hyperparameters
python RL/train.py --timesteps 25000 --batch-size 64 --lr 0.0003
```

### Visual Evaluation
Evaluate the trained policy with the interactive MuJoCo visualization window:
```bash
# Visual evaluation (opens MuJoCo viewer window)
python RL/evaluate.py --episodes 5

# Headless evaluation (console metrics only)
python RL/evaluate.py --headless --episodes 10
```
