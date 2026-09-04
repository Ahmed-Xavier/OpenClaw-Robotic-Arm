"""
robot_api.py — Robot Abstraction Layer for the MuJoCo-simulated Aubo i5 arm.

This module exposes a small, hardware-independent API:

    move_to(x, y, z, quat=None)
    pick()
    place(x, y, z)
    gripper(value)
    get_state()

It is meant to be imported directly (for testing / manual scripting) and
later wrapped by a thin CLI (arm_api.py) that OpenClaw calls as a subprocess,
the same pattern used by AI-Robotic-Arm-RDK-S100-OpenClaw's arm_api.py.

Everything here was built from lessons learned debugging the previous
Move_test.py script:
  - The eef_site is found by substring match ('eef' in name), not an exact
    name, because dm_control prefixes it (e.g. 'aubo_i5/eef_site').
  - The fingertip-to-eef offset is MEASURED from the model at startup, not
    guessed, and a fallback constant is only used with an explicit warning.
  - The cube's position is read live from physics every call, never cached
    or hardcoded, because prior attempts to guess it silently failed.
"""

import time
import numpy as np
import gymnasium
import manipulator_mujoco  # noqa: F401  (registers the gym env)

ENV_ID = "manipulator_mujoco/AuboI5Env-v0"
DOWNWARD_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
GRIP_CLEARANCE = 0.025  # meters of clearance above the cube center
FALLBACK_FINGERTIP_OFFSET = 0.165  # only used if measurement fails


class RobotAPI:
    def __init__(self, render=True):
        self.env = gymnasium.make(ENV_ID, render_mode="human" if render else None)
        self.obs, self.info = self.env.reset()
        self.physics = self.env.unwrapped._physics

        self._eef_site_id = self._find_eef_site()
        self._cube_body_id = self._find_cube_body()
        self._fingertip_offset = self._measure_fingertip_offset()

        self._holding = False
        self._last_action = np.array(
            [0.45, 0.0, 0.40, *DOWNWARD_QUAT, 0.0], dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # Startup diagnostics — run once, fail loudly instead of guessing
    # ------------------------------------------------------------------ #

    def _find_eef_site(self):
        """Find the end-effector site by substring match (name is prefixed
        by dm_control, e.g. 'aubo_i5/eef_site'), not an exact match."""
        for i in range(self.physics.model.nsite):
            name = self.physics.model.id2name(i, "site")
            if name and "eef" in name.lower():
                return i
        raise RuntimeError(
            "Could not find an eef_site in this model. "
            "Check physics.model.nsite names manually."
        )

    def _find_cube_body(self):
        """Find the graspable cube body. Falls back to env.unwrapped._box
        if the body name search finds nothing (dm_control sometimes assigns
        the default name 'unnamed_model/' to unlabeled Primitives)."""
        for i in range(self.physics.model.nbody):
            name = self.physics.model.id2name(i, "body")
            if name and any(k in name.lower() for k in ("box", "cube", "block")):
                return i
        if hasattr(self.env.unwrapped, "_box"):
            print(
                "[robot_api] WARNING: cube not found by name search, "
                "falling back to env.unwrapped._box — verify this is correct."
            )
            return self.physics.bind(self.env.unwrapped._box.geom).element_id
        raise RuntimeError(
            "No graspable cube body found in this environment. "
            "Run the unfiltered body dump to confirm one exists."
        )

    def _measure_fingertip_offset(self):
        """Measure the real vertical gap between the eef_site and the
        lowest finger/pad body. Never silently trust the fallback."""
        eef_z = self.physics.data.site_xpos[self._eef_site_id][2]
        finger_zs = []
        for i in range(self.physics.model.nbody):
            name = self.physics.model.id2name(i, "body")
            if name and ("finger" in name.lower() or "pad" in name.lower()):
                finger_zs.append(self.physics.data.xpos[i][2])

        if finger_zs:
            offset = float(abs(eef_z - min(finger_zs)))
            print(f"[robot_api] Measured fingertip-to-eef offset: {offset:.4f} m")
            return offset

        print(
            f"[robot_api] WARNING: no finger/pad bodies found — falling back "
            f"to guessed offset {FALLBACK_FINGERTIP_OFFSET:.4f} m. "
            f"Grasp height will likely be wrong until this is fixed."
        )
        return FALLBACK_FINGERTIP_OFFSET

    # ------------------------------------------------------------------ #
    # Public robot API
    # ------------------------------------------------------------------ #

    def get_state(self):
        """Return a JSON-serializable snapshot of the robot's state."""
        cube_pos = self.physics.data.xpos[self._cube_body_id].copy()
        eef_pos = self.physics.data.site_xpos[self._eef_site_id].copy()
        return {
            "holding_cube": self._holding,
            "eef_position": eef_pos.tolist(),
            "cube_position": cube_pos.tolist(),
        }

    def move_to(self, x, y, z, quat=None, gripper_value=None, steps=200):
        """Move the end-effector to an absolute [x, y, z] world position.
        Blocks until `steps` simulation steps have run (roughly enough
        time for the OSC controller to converge for small movements)."""
        if quat is None:
            quat = DOWNWARD_QUAT
        if gripper_value is None:
            gripper_value = self._last_action[7]

        action = np.array([x, y, z, *quat, gripper_value], dtype=np.float32)
        self._last_action = action
        for _ in range(steps):
            self.obs, _, _, _, self.info = self.env.step(action)
        return self.get_state()

    def gripper(self, value):
        """Set gripper openness: 0.0 = fully open, 1.0 = fully closed."""
        value = float(np.clip(value, 0.0, 1.0))
        self._last_action[7] = value
        for _ in range(50):
            self.obs, _, _, _, self.info = self.env.step(self._last_action)
        self._holding = value > 0.5
        return self.get_state()

    def pick(self, hover_height=0.15, settle_steps=100):
        """Pick up the cube using its LIVE position, not a cached one.
        This is the exact bug we fixed in Move_test.py — re-reading the
        cube's position on every call instead of trusting a stale value."""
        cube_pos = self.physics.data.xpos[self._cube_body_id].copy()
        x, y = cube_pos[0], cube_pos[1]
        z_grasp = cube_pos[2] + self._fingertip_offset + GRIP_CLEARANCE
        z_hover = z_grasp + hover_height

        self.move_to(x, y, z_hover, gripper_value=0.0)
        self.move_to(x, y, z_grasp, gripper_value=0.0)
        self.gripper(1.0)
        self.move_to(x, y, z_hover, gripper_value=1.0)
        return self.get_state()

    def place(self, x, y, z, hover_height=0.15):
        """Place whatever is held at an absolute [x, y, z] position."""
        z_hover = z + hover_height
        self.move_to(x, y, z_hover, gripper_value=1.0)
        self.move_to(x, y, z, gripper_value=1.0)
        self.gripper(0.0)
        self.move_to(x, y, z_hover, gripper_value=0.0)
        return self.get_state()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # Manual smoke test — run this file directly to sanity check the API
    # before wiring it into an OpenClaw skill.
    robot = RobotAPI(render=True)
    print("Initial state:", robot.get_state())

    input("Press Enter to run pick()...")
    print("After pick:", robot.pick())

    input("Press Enter to run place(0.35, 0.25, 0.05)...")
    print("After place:", robot.place(0.35, 0.25, 0.05))

    input("Press Enter to close...")
    robot.close()
