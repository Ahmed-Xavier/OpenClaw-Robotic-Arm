"""grasp_env.py — Gymnasium wrapper for robosuite UR5e grasping the cube.

Phase 2 RL Environment:
- Underlying robot: UR5e arm in robosuite Lift environment
- Goal: Move UR5e end-effector (EEF) to the cube and close the gripper to achieve a grasp
- Observation: 10-dimensional vector [eef_pos(3), cube_pos(3), cube_pos - eef_pos(3), gripper_state(1)]
- Action: Continuous 4D [dx, dy, dz, gripper] mapped into 7D robosuite OSC action
- Reward: Dense distance progress + proximity penalty + action penalty + contact bonus + grasp bonus
- Success: robosuite _check_grasp returns True (both fingerpads contact the cube simultaneously)
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

GRASP_BONUS: float = 10.0       # Reward for successful grasp (both fingerpads contact cube)
CONTACT_BONUS: float = 0.25     # Per-step reward for any finger-cube contact
ACTION_PENALTY_COEF: float = 0.01


class GraspEnv(gym.Env):
    """Gymnasium environment wrapping robosuite's UR5e Lift environment for grasping.

    The policy controls 3D Cartesian translation AND the gripper open/close command.
    Success requires robosuite's _check_grasp to return True, meaning both
    left and right fingerpads of the Robotiq85 gripper simultaneously contact the cube.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 20,
    }

    def __init__(
        self,
        has_renderer: bool = False,
        control_freq: int = 20,
        max_episode_steps: int = 150,
        success_threshold: float = 0.03,  # Distance threshold for reference/info only
    ):
        super().__init__()
        self.has_renderer = has_renderer
        self.control_freq = control_freq
        self.max_episode_steps = max_episode_steps
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

        # Action space: 4D [dx, dy, dz, gripper] in [-1, 1]
        # gripper: -1 = open, +1 = close
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation space: 10-dimensional vector
        # [eef_pos(3), cube_pos(3), cube_pos - eef_pos(3), gripper_state(1)]
        # gripper_state is the first driver joint qpos (range approx [-0.05, 0.41])
        obs_limit = 5.0
        self.observation_space = spaces.Box(
            low=-np.ones(10, dtype=np.float32) * obs_limit,
            high=np.ones(10, dtype=np.float32) * obs_limit,
            shape=(10,),
            dtype=np.float32,
        )

        # Episode state variables
        self.prev_distance: float = 0.0
        self.step_count: int = 0

    def _get_obs(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """Extract compact 10D observation vector.

        [eef_pos(3), cube_pos(3), cube_pos - eef_pos(3), gripper_state(1)]

        gripper_state: first element of robot0_gripper_qpos (Robotiq85 driver joint).
        Approximately -0.05 when fully open, +0.41 when fully closed.
        """
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        cube_pos = np.asarray(raw_obs["cube_pos"], dtype=np.float32)
        eef_to_cube = (cube_pos - eef_pos).astype(np.float32)
        gripper_qpos = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)
        # Use first driver joint as scalar gripper state
        gripper_state = gripper_qpos[0:1]

        obs = np.concatenate([eef_pos, cube_pos, eef_to_cube, gripper_state], dtype=np.float32)
        return obs

    def _has_any_contact(self) -> Tuple[bool, bool]:
        """Check for finger-cube contact.

        Returns:
            (any_contact, both_sides_contact):
                any_contact: True if any finger geom (left or right) touches cube
                both_sides_contact: True if both left AND right finger groups touch cube
        """
        robot = self.env.robots[0]
        gr = robot.gripper["right"]
        left_contact = self.env.check_contact(
            gr.important_geoms["left_finger"], self.env.cube
        )
        right_contact = self.env.check_contact(
            gr.important_geoms["right_finger"], self.env.cube
        )
        return (left_contact or right_contact), (left_contact and right_contact)

    def _is_grasped(self) -> bool:
        """Check robosuite's strict grasp criterion.

        Returns True only when BOTH left_fingerpad AND right_fingerpad geom groups
        are simultaneously in contact with the cube.
        """
        robot = self.env.robots[0]
        return self.env._check_grasp(robot.gripper, self.env.cube)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset robosuite simulation and return initial observation."""
        super().reset(seed=seed)
        self.step_count = 0

        # Reset underlying robosuite Lift environment
        raw_obs = self.env.reset()

        # Read initial state
        initial_eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        cube_pos = np.asarray(raw_obs["cube_pos"], dtype=np.float32)

        # Initial distance from EEF to cube
        self.prev_distance = float(np.linalg.norm(cube_pos - initial_eef))

        obs = self._get_obs(raw_obs)
        info = {
            "distance": self.prev_distance,
            "success": False,
            "is_grasped": False,
            "has_contact": False,
            "eef_position": initial_eef.tolist(),
            "cube_position": cube_pos.tolist(),
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
            act = np.zeros(4, dtype=np.float32)

        # Construct full 7D robosuite action:
        # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:3] = np.clip(act[:3], -1.0, 1.0)   # translation
        full_action[3:6] = 0.0                            # neutral orientation delta
        full_action[6] = np.clip(act[3], -1.0, 1.0)      # gripper: -1=open, +1=close

        # Step underlying simulation
        raw_obs, _, _, _ = self.env.step(full_action)

        # Extract current state
        eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        cube_pos = np.asarray(raw_obs["cube_pos"], dtype=np.float32)
        current_distance = float(np.linalg.norm(cube_pos - eef_pos))

        # ---------------------------------------------------------------
        # Reward formulation
        # ---------------------------------------------------------------

        # 1. Distance improvement (potential-based shaping, same as Phase 1.5)
        dist_improvement = (self.prev_distance - current_distance) * 10.0

        # 2. Continuous distance penalty (encourages reaching quickly)
        proximity_penalty = -current_distance * 0.1

        # 3. Action smoothness penalty (translation only, not gripper)
        action_penalty = -ACTION_PENALTY_COEF * float(np.sum(np.square(full_action[:3])))

        reward = dist_improvement + proximity_penalty + action_penalty

        # 4. Finger contact bonus (any finger geom touching cube)
        any_contact, both_contact = self._has_any_contact()
        if any_contact:
            reward += CONTACT_BONUS

        # 5. Grasp success check (strict: both fingerpads must contact simultaneously)
        is_grasped = self._is_grasped()
        if is_grasped:
            reward += GRASP_BONUS
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
            "success": is_grasped,
            "is_grasped": is_grasped,
            "has_contact": any_contact,
            "has_both_contact": both_contact,
            "eef_position": eef_pos.tolist(),
            "cube_position": cube_pos.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the MuJoCo visualization window."""
        return self.env.render()

    def close(self):
        """Clean up simulation resources."""
        return self.env.close()


def sanity_check() -> bool:
    """Verify GraspEnv initialization, spaces, reset, step, contact and grasp detection."""
    print("[GraspEnv Sanity Check] Initializing headless GraspEnv...")
    env = GraspEnv(has_renderer=False)

    # 1. Check spaces
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space:      {env.action_space}")
    assert env.observation_space.shape == (10,), f"Observation shape must be (10,), got {env.observation_space.shape}"
    assert env.action_space.shape == (4,), f"Action shape must be (4,), got {env.action_space.shape}"

    # 2. Check reset
    obs, info = env.reset(seed=42)
    eef_pos = np.array(info["eef_position"], dtype=np.float32)
    cube_pos = np.array(info["cube_position"], dtype=np.float32)
    initial_dist = info["distance"]

    print(f"  EEF Position:      {eef_pos}")
    print(f"  Cube Position:     {cube_pos}")
    print(f"  Initial Distance:  {initial_dist:.4f} m ({initial_dist*100:.2f} cm)")
    print(f"  Initial Obs:       {obs}")
    print(f"  Obs[9] (gripper):  {obs[9]:.4f}")

    assert obs.shape == (10,), "Reset observation must have shape (10,)"
    assert not np.any(np.isnan(obs)), "Observation contains NaN!"
    assert not np.any(np.isinf(obs)), "Observation contains Inf!"

    # 3. Check step with random action (gripper open)
    action_open = np.array([0.5, 0.0, -0.5, -1.0], dtype=np.float32)
    print(f"\n  Step 1 Action (open gripper): {action_open}")
    next_obs, reward, terminated, truncated, step_info = env.step(action_open)
    print(f"  Step Output -> Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
    print(f"  Step Info:   distance={step_info['distance']:.4f}, contact={step_info['has_contact']}, grasped={step_info['is_grasped']}")

    step_eef = np.array(step_info["eef_position"], dtype=np.float32)
    assert not np.allclose(eef_pos, step_eef), "EEF position should change after action!"
    assert next_obs.shape == (10,), "Next observation must have shape (10,)"
    assert not np.any(np.isnan(next_obs)), "Step observation contains NaN!"
    assert np.isfinite(reward), "Reward is not finite!"

    # 4. Check step with gripper close action
    action_close = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    print(f"\n  Step 2 Action (close gripper): {action_close}")
    obs2, reward2, _, _, info2 = env.step(action_close)
    print(f"  Step Output -> Reward: {reward2:.4f}, Obs[9] (gripper): {obs2[9]:.4f}")
    assert obs2.shape == (10,), "Observation shape must remain (10,)"

    # 5. Verify contact/grasp detection methods are callable
    any_c, both_c = env._has_any_contact()
    grasped = env._is_grasped()
    print(f"\n  Contact detection: any={any_c}, both_sides={both_c}")
    print(f"  Grasp detection:   {grasped}")

    # 6. Run a few more steps to verify stability
    for i in range(5):
        action = env.action_space.sample()
        obs_i, r_i, term_i, trunc_i, info_i = env.step(action)
        assert obs_i.shape == (10,), f"Observation shape changed at step {i+3}!"
        assert np.isfinite(r_i), f"Reward not finite at step {i+3}!"

    env.close()
    print("\n[GraspEnv Sanity Check] SUCCESS! All checks passed.\n")
    return True


if __name__ == "__main__":
    sanity_check()
