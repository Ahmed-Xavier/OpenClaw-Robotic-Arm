"""reach_env.py — Minimal Gymnasium wrapper for robosuite UR5e end-effector reaching.

Phase 1 RL Environment:
- Underlying robot: UR5e arm in robosuite Lift environment
- Goal: Move UR5e end-effector (EEF) to a randomly sampled 3D target position
- Observation: 9-dimensional vector [eef_pos(3), target_pos(3), eef_to_target(3)]
- Action: Continuous 3D Cartesian translation deltas (dx, dy, dz) mapped into 7D robosuite OSC action
- Reward: Dense distance progress + proximity reward + success bonus
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

SUCCESS_THRESHOLD: float = 0.03  # 3 cm reaching precision
SUCCESS_BONUS: float = 5.0
ACTION_PENALTY_COEF: float = 0.01

# Conservative workspace bounds around table/robot for valid reachable targets
WORKSPACE_BOUNDS = {
    "x": (-0.35, 0.05),
    "y": (-0.20, 0.20),
    "z": (0.86, 1.10),  # Table is at z=0.80 m
}


class ReachEnv(gym.Env):
    """Gymnasium environment wrapping robosuite's UR5e Lift environment for reaching."""

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

        # Instantiate robosuite Lift environment
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
        # (or 7D if explicitly requested; 3D is recommended for Phase 1)
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
        # [eef_pos(3), target_pos(3), eef_to_target(3)]
        # Finite bounds [-5.0, 5.0] and dtype float32
        obs_limit = 5.0
        self.observation_space = spaces.Box(
            low=-np.ones(9, dtype=np.float32) * obs_limit,
            high=np.ones(9, dtype=np.float32) * obs_limit,
            shape=(9,),
            dtype=np.float32,
        )

        # Episode state variables
        self.target_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.prev_distance: float = 0.0
        self.step_count: int = 0
        self._np_random: np.random.Generator = np.random.default_rng()

    def _get_obs(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """Extract compact 9D observation vector from robosuite dictionary."""
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        target_pos = self.target_pos.copy()
        eef_to_target = (target_pos - eef_pos).astype(np.float32)

        obs = np.concatenate([eef_pos, target_pos, eef_to_target], dtype=np.float32)
        return obs

    def _sample_target(self, current_eef_pos: np.ndarray) -> np.ndarray:
        """Sample a reachable target near current EEF within conservative workspace."""
        # Relative perturbation around current EEF position
        delta = self._np_random.uniform(
            low=[-0.12, -0.12, -0.08],
            high=[0.12, 0.12, 0.10],
            size=(3,),
        ).astype(np.float32)

        # Ensure minimum initial distance so target is not trivially reached at step 0
        if np.linalg.norm(delta) < 0.05:
            delta += np.array([0.06, 0.0, 0.0], dtype=np.float32)

        target = current_eef_pos + delta

        # Clip strictly to safe workspace envelope
        target[0] = np.clip(target[0], WORKSPACE_BOUNDS["x"][0], WORKSPACE_BOUNDS["x"][1])
        target[1] = np.clip(target[1], WORKSPACE_BOUNDS["y"][0], WORKSPACE_BOUNDS["y"][1])
        target[2] = np.clip(target[2], WORKSPACE_BOUNDS["z"][0], WORKSPACE_BOUNDS["z"][1])

        return target.astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset robosuite simulation, sample a new target, and return initial observation."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.step_count = 0

        # Reset underlying robosuite environment
        raw_obs = self.env.reset()

        # Read initial EEF position and sample reachable target
        initial_eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        self.target_pos = self._sample_target(initial_eef)

        # Initial distance
        self.prev_distance = float(np.linalg.norm(initial_eef - self.target_pos))

        obs = self._get_obs(raw_obs)
        info = {
            "distance": self.prev_distance,
            "success": False,
            "target": self.target_pos.tolist(),
            "eef_position": initial_eef.tolist(),
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
        full_action[6] = -1.0    # neutral open gripper

        # Step underlying simulation
        raw_obs, _, _, _ = self.env.step(full_action)

        # Extract current state
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        current_distance = float(np.linalg.norm(eef_pos - self.target_pos))

        # Reward formulation:
        # 1. Distance improvement (potential-based shaping)
        dist_improvement = (self.prev_distance - current_distance) * 10.0
        # 2. Continuous distance penalty (encourages reaching quickly)
        proximity_penalty = -current_distance * 0.1
        # 3. Action smoothness penalty
        action_penalty = -ACTION_PENALTY_COEF * float(np.sum(np.square(full_action[:3])))

        reward = dist_improvement + proximity_penalty + action_penalty

        # Check success condition
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
            "target": self.target_pos.tolist(),
            "eef_position": eef_pos.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the MuJoCo visualization window."""
        return self.env.render()

    def close(self):
        """Clean up simulation resources."""
        return self.env.close()


def sanity_check() -> bool:
    """Verify ReachEnv initialization, observation/action specs, reset, and step."""
    print("[ReachEnv Sanity Check] Initializing headless ReachEnv...")
    env = ReachEnv(has_renderer=False, action_dim=3)

    # 1. Check spaces
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space:      {env.action_space}")
    assert env.observation_space.shape == (9,), "Observation shape must be (9,)"
    assert env.action_space.shape == (3,), "Action shape must be (3,)"

    # 2. Check reset
    obs, info = env.reset(seed=42)
    print(f"  Initial Obs:       {obs}")
    print(f"  Initial Info:      {info}")
    assert obs.shape == (9,), "Reset observation must have shape (9,)"
    assert not np.any(np.isnan(obs)), "Observation contains NaN!"
    assert not np.any(np.isinf(obs)), "Observation contains Inf!"

    # 3. Check step with random action
    action = env.action_space.sample()
    print(f"  Sampled Action:    {action}")
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    print(f"  Step Output -> Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
    print(f"  Step Info:   {step_info}")
    assert next_obs.shape == (9,), "Next observation must have shape (9,)"
    assert not np.any(np.isnan(next_obs)), "Step observation contains NaN!"

    env.close()
    print("[ReachEnv Sanity Check] SUCCESS! All checks passed.\n")
    return True


if __name__ == "__main__":
    sanity_check()
