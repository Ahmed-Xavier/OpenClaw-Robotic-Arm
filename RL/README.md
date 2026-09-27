# UR5e Robotic Manipulation via Reinforcement Learning

This directory contains the reinforcement learning system for robotic manipulation in the OpenClaw repository.

---

## Roadmap & Architecture Principle

```text
Phase 1: Reach Arbitrary Target (ReachEnv)          [COMPLETE]
                    ↓
Phase 1.5: Reach Actual Physical Cube (ObjectReachEnv) [COMPLETE]
                    ↓
Phase 2: Grasp Cube (Contact & Gripper Control)       [IMPLEMENTED / SMOKE-TESTED]
                    ↓
Phase 3: Lift & Object Transport                      [PLANNED]
```

### Control Stack Integration

```text
High-Level Planner / LLM (Future)
        ↓
High-Level Skill Layer (pick, place, push)
        ↓
RL Policy (Low-level learned skill: GraspEnv / ObjectReachEnv / ReachEnv)
        ↓
Operational Space Controller (OSC)
        ↓
MuJoCo Physics Simulation / Physical Robot
```

**Key Isolation Guarantees**:
- Completely independent of LLMs, prompts, and tool calling.
- Does not modify or depend on [`robot_api.py`](../robot_api.py).
- Completely isolated under `RL/`.

---

## Phase 1.5: UR5e Physical Cube Reaching (`ObjectReachEnv`)

Phase 1.5 transitions the reaching objective from a synthetic 3D Cartesian point to the **real physical cube** spawned on the tabletop in robosuite's `Lift` environment.

### 1. Environment Design (`ObjectReachEnv`)

- **File**: [`RL/envs/object_reach_env.py`](envs/object_reach_env.py)
- **Simulation Engine**: `robosuite 1.5.2` (`Lift` environment with `UR5e` robot).
- **Object Observations**: Enabled (`use_object_obs=True`).
- **Control Mode**: Operational Space Control (`OSC_POSE`) at `control_freq=20` Hz (50 ms per step).
- **Episode Horizon**: 100 steps (5.0 seconds simulation time).
- **Reset Behavior**: Robosuite spawns the physical cube randomly across the table plane ($Z \approx 0.81\text{--}0.83$ m, $X \approx [-0.05, 0.05]$ m, $Y \approx [-0.05, 0.05]$ m).

### 2. Observation Space
A compact 9-dimensional numerical vector (`dtype=float32`):
| Indices | Dimension | Description | Finite Bounds |
| :--- | :--- | :--- | :--- |
| `0:3` | 3 | Current EEF position $\mathbf{p}_{\text{eef}} = (x, y, z)$ | $[-5.0, 5.0]$ |
| `3:6` | 3 | Real Cube position $\mathbf{p}_{\text{cube}} = (x_c, y_c, z_c)$ | $[-5.0, 5.0]$ |
| `6:9` | 3 | Relative displacement $\mathbf{p}_{\text{cube}} - \mathbf{p}_{\text{eef}}$ | $[-5.0, 5.0]$ |

*Robosuite Relative Vector Verification*: In robosuite `Lift`, `raw_obs["gripper_to_cube_pos"]` identically equals $\mathbf{p}_{\text{cube}} - \mathbf{p}_{\text{eef}}$. The policy observation explicitly computes $\mathbf{p}_{\text{cube}} - \mathbf{p}_{\text{eef}}$ to keep semantics identical to Phase 1.

### 3. Action Space
- **Policy Action Space**: Continuous 3D Cartesian delta translations:
  $$\mathbf{a}_t = [dx, dy, dz] \in [-1.0, 1.0]^3$$
- **Action Mapping**: Mapped to the 7-dimensional robosuite OSC action:
  ```python
  full_action = [dx, dy, dz, 0.0, 0.0, 0.0, -1.0]
  #                  ^           ^              ^
  #             Translation   Neutral Ori    Open Gripper
  ```
  *(Grasping is deliberately disabled in Phase 1.5; gripper is kept open).*

### 4. Reward Function
$$r_t = 10 \cdot (d_{t-1} - d_t) - 0.1 \cdot d_t - 0.01 \|\mathbf{a}_t\|^2 + r_{\text{bonus}}$$
where:
- $d_t = \|\mathbf{p}_{\text{eef}} - \mathbf{p}_{\text{cube}}\|$ is Euclidean distance to the physical cube.
- $10 \cdot (d_{t-1} - d_t)$ awards potential-based distance improvement.
- $-0.1 \cdot d_t$ continuous proximity penalty.
- $-0.01 \|\mathbf{a}_t\|^2$ action penalty for smoothness.
- $r_{\text{bonus}} = +5.0$ awarded upon success.

### 5. Success Condition
An episode terminates immediately as **SUCCESS** when:
$$\|\mathbf{p}_{\text{eef}} - \mathbf{p}_{\text{cube}}\| < 0.03 \text{ m} \quad (3 \text{ cm})$$

### 6. Training Configuration
- **Algorithm**: PPO (Stable-Baselines3 `MlpPolicy`)
- **Timesteps**: 50,000 steps
- **Rollout buffer (`n_steps`)**: 1024
- **Mini-batch size**: 64
- **Learning rate**: $3 \times 10^{-4}$
- **Discount factor ($\gamma$)**: 0.99
- **GAE ($\lambda$)**: 0.95
- **Weights Path**: [`RL/models/ur5e_object_reach_ppo.zip`](models/ur5e_object_reach_ppo.zip)

### 7. Evaluation Results (20 Test Episodes)

| Metric | Result |
| :--- | :--- |
| **Success Rate** | **20 / 20 (100.0%)** |
| **Average Final Distance** | **2.34 cm** |
| **Best Final Distance** | **1.86 cm** |
| **Average Episode Length** | **20.4 steps** (~1.02 seconds) |
| **Initial Distance Range** | **[25.10 cm, 31.26 cm]** |

---

## Comparison: Phase 1 vs. Phase 1.5

| Dimension | Phase 1 (`ReachEnv`) | Phase 1.5 (`ObjectReachEnv`) |
| :--- | :--- | :--- |
| **Target Type** | Synthetic Cartesian point $(x_t, y_t, z_t)$ | Physical rigid cube on table |
| **Target Distribution** | Uniform random perturbation in conservative 3D box | Robosuite `Lift` physics table placement |
| **Initial Distance Range** | $\sim 5\text{--}18$ cm | $\sim 25\text{--}31$ cm |
| **Success Rate (20 eps)** | 20 / 20 (100%) | 20 / 20 (100%) |
| **Average Final Distance** | 2.35 cm | 2.34 cm |
| **Average Steps to Success** | ~14.2 steps | ~20.4 steps |
| **Training Stability** | High, steady convergence | High, steady convergence |
| **Model Checkpoint** | `RL/models/ur5e_reach_ppo.zip` | `RL/models/ur5e_object_reach_ppo.zip` |

### Key Observations
1. **Initial Distance & Trajectory Length**: In Phase 1, target points were sampled close to the reset pose ($\sim 5\text{--}18$ cm). In Phase 1.5, the physical cube rests on the table, resulting in larger initial displacements ($\sim 25\text{--}31$ cm). Despite this, the policy converged reliably and reaches the 3 cm threshold in an average of 20.4 steps (approx. 1.02 s of wall-clock simulation time).
2. **Terminal Precision**: Both policies achieve consistent sub-3 cm precision (2.35 cm vs 2.34 cm average), confirming that conditioning on the physical cube coordinates transfers with high accuracy.

---

## Phase 1 Reference: UR5e End-Effector Reaching (`ReachEnv`)

- **File**: [`RL/envs/reach_env.py`](envs/reach_env.py)
- **Goal**: Move UR5e end-effector (EEF) to a randomly sampled reachable 3D target position.
- **Model Checkpoint**: [`RL/models/ur5e_reach_ppo.zip`](models/ur5e_reach_ppo.zip)
- **Threshold**: 3 cm precision.

---

## Phase 2: UR5e Object Grasping (`GraspEnv`)

Phase 2 builds upon Phase 1.5 by granting the policy active control of the `Robotiq85` parallel-jaw gripper, teaching the robot to coordinate 3D spatial alignment with closing the gripper around the physical cube.

### 1. Environment Design (`GraspEnv`)

- **File**: [`RL/envs/grasp_env.py`](envs/grasp_env.py)
- **Simulation Engine**: `robosuite 1.5.2` (`Lift` environment with `UR5e` robot and `Robotiq85Gripper`).
- **Object Observations**: Enabled (`use_object_obs=True`).
- **Control Mode**: Operational Space Control (`OSC_POSE`) at `control_freq=20` Hz (50 ms per step).
- **Episode Horizon**: 150 steps (7.5 seconds simulation time, extended from 100 steps to accommodate approach, descent, and finger closure dynamics).
- **Reset Behavior**: Same physical tabletop distribution as Phase 1.5.

### 2. Observation Space
A 10-dimensional numerical vector (`dtype=float32`):
| Indices | Dimension | Description | Bounds |
| :--- | :--- | :--- | :--- |
| `0:3` | 3 | Current EEF position $\mathbf{p}_{\text{eef}} = (x, y, z)$ | $[-5.0, 5.0]$ |
| `3:6` | 3 | Real Cube position $\mathbf{p}_{\text{cube}} = (x_c, y_c, z_c)$ | $[-5.0, 5.0]$ |
| `6:9` | 3 | Relative displacement $\mathbf{p}_{\text{cube}} - \mathbf{p}_{\text{eef}}$ | $[-5.0, 5.0]$ |
| `9:10` | 1 | Gripper driver joint position `robot0_gripper_qpos[0]` ($\approx [-0.05, 0.41]$) | $[-5.0, 5.0]$ |

### 3. Action Space
Continuous 4D vector $\mathbf{a}_t = [dx, dy, dz, \text{gripper}] \in [-1.0, 1.0]^4$:
- $dx, dy, dz$: 3D Cartesian translation deltas.
- $\text{gripper}$: Gripper open/close command ($-1.0 = \text{open}, +1.0 = \text{close}$).
- Mapped to the 7-dimensional robosuite action:
  ```python
  full_action = [dx, dy, dz, 0.0, 0.0, 0.0, gripper]
  ```

### 4. Reward Formulation
$$r_t = 10 \cdot (d_{t-1} - d_t) - 0.1 \cdot d_t - 0.01 \|\mathbf{a}_{t,\text{trans}}\|^2 + r_{\text{contact}} + r_{\text{grasp}}$$
where:
- $10 \cdot (d_{t-1} - d_t)$: Potential-based distance progress toward the cube.
- $-0.1 \cdot d_t$: Proximity penalty encouraging swift reaching.
- $-0.01 \|\mathbf{a}_{t,\text{trans}}\|^2$: Smoothness penalty on translational actions.
- $r_{\text{contact}} = +0.25$: Step bonus when any gripper finger geom makes contact with the cube.
- $r_{\text{grasp}} = +10.0$: Terminal bonus when robosuite confirms a strict grasp.

### 5. Success Condition (Strict Grasp Verification)
Unlike reaching tasks, Phase 2 success is **not** determined by proximity. An episode terminates immediately as **SUCCESS** only when robosuite's built-in `_check_grasp(robot.gripper, env.cube)` returns `True`:
- **Criterion**: Both `left_fingerpad` and `right_fingerpad` collision groups must simultaneously make physical contact with the cube.
- One-sided contact or pushing without pad enclosure does not trigger success.

### 6. Implementation Status & Smoke Test
- **Sanity Check**: Verified (`python RL/envs/grasp_env.py`) — observation/action spaces, reset, steps, contact detection, and grasp checking operational.
- **5K Smoke Test**: Verified (`python RL/train.py --env grasp --timesteps 5000 --save-path RL/models/smoke_test_grasp_ppo`) — PPO rollout, training step, and model saving completed cleanly (~83 FPS).
- **Default Checkpoint Path**: `RL/models/ur5e_grasp_ppo.zip`
- **50K Full Training**: Pending user approval before execution.

---

## Usage Commands

### 1. Sanity Checks
Verify environment initialization, reset specs, relative vector verification, and step mechanics:
```bash
# Phase 1 sanity check
python RL/envs/reach_env.py

# Phase 1.5 sanity check
python RL/envs/object_reach_env.py

# Phase 2 sanity check
python RL/envs/grasp_env.py
```

### 2. Training
All training pipelines share a unified interface via `--env`:
```bash
# Phase 1 (ReachEnv)
python RL/train.py --env reach --timesteps 50000

# Phase 1.5 (ObjectReachEnv)
python RL/train.py --env object_reach --timesteps 50000

# Phase 2 5K Smoke Test (verified)
python RL/train.py --env grasp --timesteps 5000 --save-path RL/models/smoke_test_grasp_ppo

# Phase 2 Full Training (pending user approval)
python RL/train.py --env grasp --timesteps 50000
```

### 3. Evaluation
Evaluate policies in headless mode or with the interactive MuJoCo visualization window:
```bash
# Phase 2 Visual Evaluation (when trained)
python RL/evaluate.py --env grasp --episodes 10

# Phase 2 Headless Evaluation
python RL/evaluate.py --env grasp --headless --episodes 10

# Phase 1.5 Visual Evaluation (opens MuJoCo viewer window)
python RL/evaluate.py --env object_reach --episodes 20

# Phase 1.5 Headless Evaluation
python RL/evaluate.py --env object_reach --headless --episodes 20

# Phase 1 Visual Evaluation
python RL/evaluate.py --env reach --episodes 20
```

---

## What We Are Deliberately NOT Doing Yet

To preserve disciplined incremental progress:
- ❌ **No Lifting / Object Transport**: Cube stays on the table (deferred to Phase 3).
- ❌ **No Vision / Depth Cameras**: Pure low-dimensional Cartesian state.
- ❌ **No LLM / Agent Coupling**: No XavierClaw/Ollama integration.
- ❌ **No Domain Randomization / Sim2Real**: Pure deterministic MuJoCo dynamics.
