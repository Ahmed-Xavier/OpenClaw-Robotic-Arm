"""object_reach_env.py — Gymnasium wrapper for robosuite UR5e reaching the actual cube.

Phase 1.5 RL Environment:
- Underlying robot: UR5e arm in robosuite Lift environment
- Goal: Move UR5e end-effector (EEF) to the actual physical cube on the table
- Observation: 9-dimensional vector [eef_pos(3), cube_pos(3), cube_pos - eef_pos(3)]
- Action: Continuous 3D Cartesian translation deltas (dx, dy, dz) mapped into 7D robosuite OSC action
- Reward: Dense distance progress + proximity penalty + action smoothness penalty + success bonus
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import robosuite as suite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUCCESS_THRESHOLD: float = 0.03  # 3 cm reaching precision to the cube
SUCCESS_BONUS: float = 5.0
ACTION_PENALTY_COEF: float = 0.01


class ObjectReachEnv(gym.Env):
    """Gymnasium environment wrapping robosuite's UR5e Lift environment for object reaching."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 20,
    }

    def __init__(
        self,
        has_renderer: bool = False,
        control_freq: int = 20,
        max_episode_steps: int = 100,
        action_dim: int = 3,
        success_threshold: float = SUCCESS_THRESHOLD,
    ):
        super().__init__()
        self.has_renderer = has_renderer
        self.control_freq = control_freq
        self.max_episode_steps = max_episode_steps
        self.action_dim = action_dim
        self.success_threshold = success_threshold

        # Instantiate robosuite Lift environment with object observations enabled
        self.env = suite.make(
            env_name="Lift",
            robots="UR5e",
            has_renderer=self.has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq=self.control_freq,
        )

        # Action space: 3D delta translation [dx, dy, dz] in [-1, 1]
        if self.action_dim == 3:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32,
            )

        # Observation space: 9-dimensional vector
        # [eef_pos(3), cube_pos(3), cube_pos - eef_pos(3)]
        # Finite bounds [-5.0, 5.0] and dtype float32
        obs_limit = 5.0
        self.observation_space = spaces.Box(
            low=-np.ones(9, dtype=np.float32) * obs_limit,
            high=np.ones(9, dtype=np.float32) * obs_limit,
            shape=(9,),
            dtype=np.float32,
        )

        # Episode state variables
        self.cube_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.prev_distance: float = 0.0
        self.step_count: int = 0

    def _get_obs(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """Extract compact 9D observation vector [eef_pos, cube_pos, cube_pos - eef_pos]."""
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        cube_pos = np.asarray(raw_obs["cube_pos"], dtype=np.float32)
        # Explicitly compute relative vector from EEF to cube
        eef_to_cube = (cube_pos - eef_pos).astype(np.float32)

        obs = np.concatenate([eef_pos, cube_pos, eef_to_cube], dtype=np.float32)
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset robosuite simulation, read cube state, and return initial observation."""
        super().reset(seed=seed)
        self.step_count = 0

        # Reset underlying robosuite Lift environment (cube is placed randomly on table)
        raw_obs = self.env.reset()

        # Read EEF and cube positions
        initial_eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        self.cube_pos = np.asarray(raw_obs["cube_pos"], dtype=np.float32)

        # Initial distance from EEF to cube
        self.prev_distance = float(np.linalg.norm(self.cube_pos - initial_eef))

        obs = self._get_obs(raw_obs)
        info = {
            "distance": self.prev_distance,
            "success": False,
            "eef_position": initial_eef.tolist(),
            "cube_position": self.cube_pos.tolist(),
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action, update physics, and return (obs, reward, terminated, truncated, info)."""
        self.step_count += 1

        # Action sanitization
        act = np.asarray(action, dtype=np.float32).flatten()
        if np.any(np.isnan(act)) or np.any(np.isinf(act)):
            act = np.zeros(self.action_dim, dtype=np.float32)

        # Construct full 7D robosuite action:
        # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:3] = np.clip(act[:3], -1.0, 1.0)
        full_action[3:6] = 0.0   # neutral orientation delta
        full_action[6] = -1.0    # neutral open gripper (grasping not trained in this phase)

        # Step underlying simulation
        raw_obs, _, _, _ = self.env.step(full_action)

        # Extract current state
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        self.cube_pos = np.asarray(raw_obs["cube_pos"], dtype=np.float32)
        current_distance = float(np.linalg.norm(self.cube_pos - eef_pos))

        # Reward formulation:
        # 1. Distance improvement (potential-based shaping)
        dist_improvement = (self.prev_distance - current_distance) * 10.0
        # 2. Continuous distance penalty (encourages reaching quickly)
        proximity_penalty = -current_distance * 0.1
        # 3. Action smoothness penalty
        action_penalty = -ACTION_PENALTY_COEF * float(np.sum(np.square(full_action[:3])))

        reward = dist_improvement + proximity_penalty + action_penalty

        # Check success condition (reaching within 3 cm threshold)
        success = current_distance < self.success_threshold
        if success:
            reward += SUCCESS_BONUS
            terminated = True
            truncated = False
        elif self.step_count >= self.max_episode_steps:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        self.prev_distance = current_distance
        obs = self._get_obs(raw_obs)

        info = {
            "distance": current_distance,
            "success": success,
            "eef_position": eef_pos.tolist(),
            "cube_position": self.cube_pos.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the MuJoCo visualization window."""
        return self.env.render()

    def close(self):
        """Clean up simulation resources."""
        return self.env.close()


def sanity_check() -> bool:
    """Verify ObjectReachEnv initialization, observation/action specs, reset, step, and relative vector."""
    print("[ObjectReachEnv Sanity Check] Initializing headless ObjectReachEnv...")
    env = ObjectReachEnv(has_renderer=False, action_dim=3)

    # 1. Check spaces
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space:      {env.action_space}")
    assert env.observation_space.shape == (9,), "Observation shape must be (9,)"
    assert env.action_space.shape == (3,), "Action shape must be (3,)"

    # 2. Check reset
    obs, info = env.reset(seed=42)
    eef_pos = np.array(info["eef_position"], dtype=np.float32)
    cube_pos = np.array(info["cube_position"], dtype=np.float32)
    computed_relative = cube_pos - eef_pos
    raw_obs_reset = env.env._get_observations()
    robosuite_relative = np.asarray(raw_obs_reset["gripper_to_cube_pos"], dtype=np.float32)

    print(f"  EEF Position:               {eef_pos}")
    print(f"  Cube Position:              {cube_pos}")
    print(f"  Computed (cube - eef):      {computed_relative}")
    print(f"  Robosuite gripper_to_cube:  {robosuite_relative}")
    print(f"  Initial Distance:           {info['distance']:.4f} m ({info['distance']*100:.2f} cm)")
    print(f"  Initial Obs:                {obs}")

    # Relative-vector verification:
    # In robosuite Lift, raw_obs["gripper_to_cube_pos"] is defined as cube_pos - eef_pos.
    # The vectors match identically in both magnitude and direction (+ sign from EEF to cube).
    assert np.allclose(computed_relative, robosuite_relative, atol=1e-5), (
        f"Relative vector mismatch! computed={computed_relative}, robosuite={robosuite_relative}"
    )
    print("  [Relative Vector Verification] Confirmed: computed (cube_pos - eef_pos) == robosuite gripper_to_cube_pos")

    assert obs.shape == (9,), "Reset observation must have shape (9,)"
    assert not np.any(np.isnan(obs)), "Observation contains NaN!"
    assert not np.any(np.isinf(obs)), "Observation contains Inf!"

    # 3. Check step with random action
    action = env.action_space.sample()
    print(f"\n  Sampled Action:    {action}")
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    print(f"  Step Output -> Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
    print(f"  Step Info:   {step_info}")

    step_eef = np.array(step_info["eef_position"], dtype=np.float32)
    step_cube = np.array(step_info["cube_position"], dtype=np.float32)
    step_dist = step_info["distance"]

    # Verify EEF position changed
    assert not np.allclose(eef_pos, step_eef), "EEF position should change after action!"
    # Verify cube position is valid
    assert not np.any(np.isnan(step_cube)), "Cube position contains NaN!"
    # Verify distance is valid and finite
    assert np.isfinite(step_dist), "Distance is not finite!"
    # Verify reward is finite
    assert np.isfinite(reward), "Reward is not finite!"
    # Verify next observation shape
    assert next_obs.shape == (9,), "Next observation must have shape (9,)"
    assert not np.any(np.isnan(next_obs)), "Step observation contains NaN!"

    env.close()
    print("\n[ObjectReachEnv Sanity Check] SUCCESS! All checks passed.\n")
    return True


if __name__ == "__main__":
    sanity_check()
