"""
robot_api.py — Robot Abstraction Layer for the MuJoCo-simulated SO-ARM100 (SO-100).

Hardware-independent API for the Hugging Face LeRobot SO-100 arm:
    move_to(x, y, z, steps=80)
    pick(hover_height=0.08)
    place(x, y, z, hover_height=0.08)
    gripper(value)               # 0.0 = fully open, 1.0 = fully closed
    get_state()                  # Full debug snapshot (joints, sim time, etc.)
    get_semantic_state()         # Concise LLM-facing snapshot
    capture_image()              # Returns RGB array from gripper camera
    save_camera_image()          # Saves camera image to disk

All physical-action methods return a structured result contract:
    {
        "success": True,
        "action": "<name>",
        "result": { ... },
        "error": None
    }
or on failure:
    {
        "success": False,
        "action": "<name>",
        "result": None,          # may include partial diagnostics
        "error": { "code": "...", "message": "..." }
    }

RobotAPI is the final physical safety authority.  Safety checks cannot be
bypassed by calling Flask directly — they live here, not only in the agent.
"""

import os
import sys
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "third_party",
    "so_arm100",
    "interactive_scene.xml",
)

# Gripper actuator limits (Position actuator: 1.5 rad = open, 0.0 rad = closed)
JAW_OPEN = 1.5
JAW_CLOSED = 0.0

# Natural home pose for the SO-100 5-DOF arm + gripper
# Home pose: arm raised clear of the table (EEF ≈ [0, -0.347, 0.202])
# The old pose [0, -1.38, 1.79, 1.34, 0, OPEN] placed the gripper directly on
# the cube, causing the 50-step physics settle to fling objects away.
HOME_QPOS = np.array([0.0, -1.8, 1.5, 0.8, 0.0, JAW_OPEN], dtype=np.float64)

JOINT_NAMES = ["Rotation (Base)", "Pitch (Shoulder)", "Elbow", "Wrist Pitch", "Wrist Roll", "Jaw (Gripper)"]

# Coarse reachable envelope (meters) — first-pass safety filter before IK.
# These bounds are intentionally conservative; IK verification is the real gate.
REACHABLE_ENVELOPE = {
    "x": (-0.30, 0.30),
    "y": (-0.35, -0.05),
    "z": (0.00, 0.35),
}

# move_to position-error tolerance (m).  Generous for the current simulation;
# typical successful moves achieve 0.001–0.004 m error.
MOVE_TOLERANCE_M = 0.02

# Grasp verification: minimum upward cube displacement (m) to consider
# a pick successful via the position heuristic.
GRASP_LIFT_THRESHOLD_M = 0.025


class RobotAPI:
    def __init__(self, render=True, camera_width=640, camera_height=480):
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"SO-100 model scene not found at {_MODEL_PATH}")

        self.model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        self._site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube"
        )
        self._sphere_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_sphere"
        )

        # Reset to home configuration
        self.data.qpos[:6] = HOME_QPOS
        self.data.ctrl[:6] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

        # Settle simulation
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._holding = False
        self._render = render
        self._viewer = None
        self._running = True
        self._busy_moving = False

        # Offscreen camera renderer
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._renderer = mujoco.Renderer(self.model, height=camera_height, width=camera_width)

        if self._render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._sync()
            # Background keep-alive loop: viewer stays responsive for Ctrl+Click perturbations
            self._sim_thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
            self._sim_thread.start()

        print("[robot_api] SO-100 arm initialized and ready.")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _sync(self):
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _keep_alive_loop(self):
        """Continuously steps physics and syncs viewer when the arm is idle,
        allowing real-time mouse dragging (Ctrl + Right-Click) and physics interactions."""
        dt = self.model.opt.timestep
        while self._running:
            if not self._busy_moving and self._viewer is not None and self._viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                self._sync()
                time.sleep(dt)
            else:
                time.sleep(0.01)

    def _solve_ik(self, target_pos, q_seed=None, max_iter=80):
        """Solve 5-DOF IK for the SO-100 arm to position the grasp site at target_pos."""
        if q_seed is None:
            theta = np.arctan2(target_pos[0], -target_pos[1])
            q = np.array([theta, -1.38, 1.79, 1.34, 0.0], dtype=np.float64)
        else:
            q = q_seed.copy()

        best_q = q.copy()
        best_dist = 1e9

        for _ in range(max_iter):
            self._ik_data.qpos[:5] = q
            mujoco.mj_forward(self.model, self._ik_data)
            curr_pos = self._ik_data.site_xpos[self._site_id]
            err = target_pos - curr_pos
            dist = np.linalg.norm(err)

            if dist < best_dist:
                best_dist = dist
                best_q = q.copy()

            if dist < 5e-4:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self._ik_data, jacp, jacr, self._site_id)
            J = jacp[:, :5]

            lambda_val = 1e-4 if dist < 0.02 else 1e-3
            dq = J.T @ np.linalg.inv(J @ J.T + lambda_val * np.eye(3)) @ err
            q += np.clip(dq * 0.4, -0.1, 0.1)

            for j in range(5):
                q[j] = np.clip(
                    q[j], self.model.jnt_range[j][0], self.model.jnt_range[j][1]
                )

        return best_q, best_dist  # return achieved IK distance too

    def _step_to_ctrl(self, target_ctrl, steps=100):
        """Smoothly interpolate actuator targets over `steps` simulation steps."""
        self._busy_moving = True
        try:
            start_ctrl = self.data.ctrl.copy()
            dt = self.model.opt.timestep

            for s in range(steps):
                alpha = (s + 1) / steps
                self.data.ctrl[:] = (1 - alpha) * start_ctrl + alpha * target_ctrl
                mujoco.mj_step(self.model, self.data)

                if self._render and s % 2 == 0:
                    self._sync()
                    time.sleep(dt * 1.2)
        finally:
            self._busy_moving = False

    # ------------------------------------------------------------------ #
    # Phase 1 — Structured result contract
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_result(action, result=None, error_code=None, error_message=None):
        """Build the standard structured result dictionary.

        Success:  _make_result("move_to", result={...})
        Failure:  _make_result("move_to", error_code="IK_FAILED", error_message="...")
        """
        success = error_code is None
        return {
            "success": success,
            "action": action,
            "result": result,
            "error": None if success else {"code": error_code, "message": error_message},
        }

    # ------------------------------------------------------------------ #
    # Phase 2 — Physical safety validation (RobotAPI is final authority)
    # ------------------------------------------------------------------ #

    def _validate_target(self, x, y, z):
        """Validate that (x, y, z) is physically reachable.

        Steps:
          1. Numeric type check.
          2. Coarse envelope check (fast rejection).
          3. IK solve and residual check (physical reachability).

        Returns:
            (ok: bool, error_code: str | None, error_message: str | None,
             ik_dist: float | None)
        """
        # 1. Type check
        try:
            x, y, z = float(x), float(y), float(z)
        except (TypeError, ValueError):
            return False, "INVALID_ARGUMENT", "Coordinates must be numeric.", None

        # 2. Coarse envelope
        (x_min, x_max) = REACHABLE_ENVELOPE["x"]
        (y_min, y_max) = REACHABLE_ENVELOPE["y"]
        (z_min, z_max) = REACHABLE_ENVELOPE["z"]

        if not (x_min <= x <= x_max):
            return (False, "OUT_OF_BOUNDS",
                    f"X={x:+.3f}m is outside reachable envelope [{x_min:.2f}, {x_max:.2f}]m.", None)
        if not (y_min <= y <= y_max):
            return (False, "OUT_OF_BOUNDS",
                    f"Y={y:+.3f}m is outside reachable envelope [{y_min:.2f}, {y_max:.2f}]m.", None)
        if not (z_min <= z <= z_max):
            return (False, "OUT_OF_BOUNDS",
                    f"Z={z:+.3f}m is outside reachable envelope [{z_min:.2f}, {z_max:.2f}]m.", None)

        # 3. IK reachability probe (uses the separate _ik_data to avoid disturbing live data)
        target_pos = np.array([x, y, z], dtype=np.float64)
        _, ik_dist = self._solve_ik(target_pos, q_seed=self.data.ctrl[:5].copy())

        if ik_dist > MOVE_TOLERANCE_M:
            return (False, "NOT_REACHABLE",
                    f"Target [{x:.3f}, {y:.3f}, {z:.3f}] is not reachable "
                    f"(IK residual {ik_dist*100:.1f} cm > tolerance {MOVE_TOLERANCE_M*100:.0f} cm).",
                    ik_dist)

        return True, None, None, ik_dist

    # ------------------------------------------------------------------ #
    # Public Robot API — Debugging / telemetry
    # ------------------------------------------------------------------ #

    def get_state(self):
        """Return a structured dictionary of live simulation telemetry (debug/full)."""
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        cube_vel = self.data.cvel[self._cube_body_id].copy()
        eef_pos = self.data.site_xpos[self._site_id].copy()
        joint_qpos = self.data.qpos[:6].tolist()
        jaw_pos = float(self.data.qpos[5])

        dist_to_cube = float(np.linalg.norm(eef_pos - cube_pos))

        st = {
            "holding_cube": self._holding,
            "dist_to_cube": dist_to_cube,
            "eef_position": {
                "x": float(eef_pos[0]),
                "y": float(eef_pos[1]),
                "z": float(eef_pos[2]),
            },
            "cube_position": {
                "x": float(cube_pos[0]),
                "y": float(cube_pos[1]),
                "z": float(cube_pos[2]),
            },
            "gripper": {
                "openness": float(np.clip((jaw_pos - JAW_CLOSED) / (JAW_OPEN - JAW_CLOSED), 0.0, 1.0)),
                "raw_rad": jaw_pos,
                "is_closed": self._holding,
            },
            "joints_rad": dict(zip(JOINT_NAMES, joint_qpos)),
            "sim_time": float(self.data.time),
        }

        if self._sphere_body_id != -1:
            sphere_pos = self.data.xpos[self._sphere_body_id].copy()
            st["sphere_position"] = {
                "x": float(sphere_pos[0]),
                "y": float(sphere_pos[1]),
                "z": float(sphere_pos[2]),
            }
            st["dist_to_sphere"] = float(np.linalg.norm(eef_pos - sphere_pos))

        return st

    def get_semantic_state(self):
        """Return a concise semantic state suitable for the LLM.

        Intentionally avoids joint angles, sim_time, and other fields that
        the LLM does not need and that waste context budget.
        """
        eef = self.data.site_xpos[self._site_id].copy()
        cube = self.data.xpos[self._cube_body_id].copy()
        jaw_pos = float(self.data.qpos[5])
        openness = float(np.clip((jaw_pos - JAW_CLOSED) / (JAW_OPEN - JAW_CLOSED), 0.0, 1.0))

        gripper_label = "open" if openness > 0.5 else "closed"
        robot_label = "holding" if self._holding else "ready"

        sem = {
            "robot": robot_label,
            "gripper": gripper_label,
            "holding": self._holding,
            "eef": [round(float(eef[0]), 4), round(float(eef[1]), 4), round(float(eef[2]), 4)],
            "cube": [round(float(cube[0]), 4), round(float(cube[1]), 4), round(float(cube[2]), 4)],
        }
        if self._sphere_body_id != -1:
            sphere = self.data.xpos[self._sphere_body_id].copy()
            sem["sphere"] = [round(float(sphere[0]), 4), round(float(sphere[1]), 4), round(float(sphere[2]), 4)]

        return sem

    def print_info(self):
        """Display cleanly formatted simulation telemetry."""
        st = self.get_state()
        eef = st["eef_position"]
        cube = st["cube_position"]
        grip = st["gripper"]

        print("\n" + "=" * 50)
        print("           SIMULATION TELEMETRY")
        print("=" * 50)
        print(f" Sim Time:       {st['sim_time']:.2f} s")
        print(f" Holding Cube:   {'[YES]' if st['holding_cube'] else '[NO]'}")
        print(f" End-Effector:   X: {eef['x']:+.4f} m | Y: {eef['y']:+.4f} m | Z: {eef['z']:+.4f} m")
        print(f" Cube Position:  X: {cube['x']:+.4f} m | Y: {cube['y']:+.4f} m | Z: {cube['z']:+.4f} m")
        print(f" Distance (EEF): {st['dist_to_cube']*100:.2f} cm")
        if "sphere_position" in st:
            sph = st["sphere_position"]
            print(f" Sphere Position:X: {sph['x']:+.4f} m | Y: {sph['y']:+.4f} m | Z: {sph['z']:+.4f} m")
            print(f" Dist to Sphere: {st['dist_to_sphere']*100:.2f} cm")
        print(f" Gripper State:  {grip['openness']*100:.1f}% Open (Jaw: {grip['raw_rad']:.3f} rad)")
        print("\n Joint Positions:")
        for name, rad in st["joints_rad"].items():
            deg = np.rad2deg(rad)
            print(f"   • {name:<18}: {rad:+.3f} rad ({deg:+.1f}°)")
        print("=" * 50 + "\n")

    def capture_image(self, camera_name="wrist_camera"):
        """Capture an RGB array from the wrist camera."""
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render()

    def save_camera_image(self, filepath="wrist_camera_view.png", camera_name="wrist_camera"):
        """Capture and save an image snapshot to disk."""
        img_arr = self.capture_image(camera_name)
        img = Image.fromarray(img_arr)
        img.save(filepath)
        print(f"[robot_api] Camera snapshot saved to: {filepath}")
        return filepath

    # ------------------------------------------------------------------ #
    # Public Robot API — Physical actions (structured results)
    # ------------------------------------------------------------------ #

    def move_to(self, x, y, z, steps=80):
        """Move the end-effector (grasp site) to an absolute [x, y, z] target.

        Phase 2: validates via _validate_target (safety cannot be bypassed).
        Phase 3: verifies actual EEF position after execution and returns
                 position_error in the structured result.
        """
        # Phase 2 — physical safety validation
        ok, err_code, err_msg, _ = self._validate_target(x, y, z)
        if not ok:
            return self._make_result("move_to", error_code=err_code, error_message=err_msg)

        target_pos = np.array([x, y, z], dtype=np.float64)

        # Solve IK using the current control state as warm-start seed
        target_q5, _ = self._solve_ik(target_pos, q_seed=self.data.ctrl[:5].copy())

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[:5] = target_q5
        self._step_to_ctrl(target_ctrl, steps=steps)

        # Phase 3 — verify achieved EEF position after motion
        actual_pos = self.data.site_xpos[self._site_id].copy()
        position_error = float(np.linalg.norm(actual_pos - target_pos))

        result_detail = {
            "requested": [round(x, 4), round(y, 4), round(z, 4)],
            "actual": [round(float(actual_pos[0]), 4),
                       round(float(actual_pos[1]), 4),
                       round(float(actual_pos[2]), 4)],
            "position_error": round(position_error, 4),
        }

        if position_error > MOVE_TOLERANCE_M:
            print(f"[robot_api] move_to: position error {position_error*100:.1f} cm exceeds tolerance.")
            return self._make_result(
                "move_to",
                result=result_detail,
                error_code="IK_FAILED",
                error_message=f"Target could not be reached within tolerance "
                              f"({position_error*100:.1f} cm > {MOVE_TOLERANCE_M*100:.0f} cm).",
            )

        return self._make_result("move_to", result=result_detail)

    def gripper(self, value, steps=60):
        """Set gripper openness: 0.0 = fully open, 1.0 = fully closed.

        Returns a structured result with the commanded and actual jaw openness.
        """
        try:
            val = float(value)
        except (TypeError, ValueError):
            return self._make_result(
                "gripper",
                error_code="INVALID_ARGUMENT",
                error_message=f"Gripper value must be a number 0.0–1.0, got {value!r}.",
            )

        if not (0.0 <= val <= 1.0):
            return self._make_result(
                "gripper",
                error_code="INVALID_ARGUMENT",
                error_message=f"Gripper value {val} is outside valid range [0.0, 1.0].",
            )

        jaw_target = (1.0 - val) * JAW_OPEN + val * JAW_CLOSED

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[5] = jaw_target
        self._step_to_ctrl(target_ctrl, steps=steps)

        # NOTE: gripper openness does NOT determine holding state.
        # self._holding is authoritative and is only set by pick() (verified
        # via position heuristic) and cleared by pick() failure / place().

        jaw_pos = float(self.data.qpos[5])
        openness = float(np.clip((jaw_pos - JAW_CLOSED) / (JAW_OPEN - JAW_CLOSED), 0.0, 1.0))

        return self._make_result("gripper", result={
            "commanded": round(val, 4),
            "openness": round(openness, 4),
            "is_closed": self._holding,
        })

    def pick(self, target="cube", approach_dist=0.06, hover_height=0.08, steps=80):
        """Pick up an object (cube or sphere) with a horizontal side approach.

        HIGH-LEVEL DETERMINISTIC SKILL.  Qwen calls pick(); all motion
        sequencing (open → approach → slide-in → settle → close → lift)
        happens deterministically here.

        Phase 4:
          - Precondition: not already holding.
          - Uses position heuristic for grasp verification (not force sensing).
          - Returns structured result with grasp_verification block.
        """
        # Precondition: must not already be holding
        if self._holding:
            held = getattr(self, "_held_object", "an object") or "an object"
            return self._make_result(
                "pick",
                error_code="INVALID_STATE",
                error_message=f"Robot is already holding {held}.",
            )

        target_str = str(target).strip().lower() if target else "cube"
        if target_str == "sphere" and self._sphere_body_id != -1:
            obj_body_id = self._sphere_body_id
            obj_name = "sphere"
        else:
            obj_body_id = self._cube_body_id
            obj_name = "cube"

        obj_pos = self.data.xpos[obj_body_id].copy()
        x, y, z = obj_pos[0], obj_pos[1], obj_pos[2]

        # Direction vector from arm base (origin) toward the object, XY only
        vec = np.array([x, y, 0.0])
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            approach_dir = np.array([0.0, -1.0, 0.0])
        else:
            approach_dir = vec / dist   # unit vector toward object

        # Standoff position: object height, approach_dist further away from the arm
        sx = x - approach_dir[0] * approach_dist
        sy = y - approach_dir[1] * approach_dist

        print(f"[robot_api] Picking {obj_name} at ({x:+.3f}, {y:+.3f}, {z:+.3f})")
        print(f"[robot_api] Side-approach standoff: ({sx:+.3f}, {sy:+.3f}, {z:+.3f})")

        # Record object Z before lifting (for grasp heuristic)
        obj_z_before = float(self.data.xpos[obj_body_id][2])

        # 1. Open gripper
        self.gripper(0.0, steps=40)

        # 2. Move to standoff at object height
        print(f"[robot_api] Moving to side standoff...")
        self.move_to(sx, sy, z, steps=steps)

        # 3. Slide in horizontally
        print(f"[robot_api] Sliding in horizontally to {obj_name}...")
        self.move_to(x, y, z, steps=steps)

        # 4. Settle
        print("[robot_api] Settling...")
        self._busy_moving = True
        try:
            settle_steps = 120
            dt = self.model.opt.timestep
            for s in range(settle_steps):
                mujoco.mj_step(self.model, self.data)
                if self._render and s % 3 == 0:
                    self._sync()
                    time.sleep(dt * 1.5)
        finally:
            self._busy_moving = False

        # 5. Close gripper
        print("[robot_api] Closing gripper...")
        self.gripper(1.0, steps=80)

        # 6. Grip dwell — 0.5 s before lifting
        print("[robot_api] Holding grip (0.5 s)...")
        self._busy_moving = True
        try:
            dwell_steps = int(0.5 / self.model.opt.timestep)
            dt = self.model.opt.timestep
            for s in range(dwell_steps):
                mujoco.mj_step(self.model, self.data)
                if self._render and s % 3 == 0:
                    self._sync()
                    time.sleep(dt)
        finally:
            self._busy_moving = False

        # 7. Lift straight up
        print("[robot_api] Lifting...")
        self.move_to(x, y, z + hover_height, steps=steps)

        # Phase 4 — Position-heuristic grasp verification
        # Check whether the object actually rose with the arm.
        obj_z_after = float(self.data.xpos[obj_body_id][2])
        obj_z_delta = obj_z_after - obj_z_before
        grasp_verified = obj_z_delta >= GRASP_LIFT_THRESHOLD_M

        print(f"[robot_api] {obj_name.capitalize()} Z: before={obj_z_before:.4f} after={obj_z_after:.4f} "
              f"delta={obj_z_delta:.4f} verified={grasp_verified}")

        grasp_verification = {
            "method": "position_heuristic",
            "verified": grasp_verified,
            "object": obj_name,
            "object_z_before": round(obj_z_before, 4),
            "object_z_after": round(obj_z_after, 4),
            "object_z_delta": round(obj_z_delta, 4),
            "threshold": GRASP_LIFT_THRESHOLD_M,
        }

        if not grasp_verified:
            # The gripper closed but the object didn't move with it — grasp failed.
            self._holding = False
            self._held_object = None
            return self._make_result(
                "pick",
                result={
                    "holding": False,
                    "object": obj_name,
                    "grasp_verification": grasp_verification,
                },
                error_code="EXECUTION_FAILED",
                error_message=f"{obj_name.capitalize()} did not move with the gripper during the lift.",
            )

        # Grasp verified — authoritative holding state set here, NOT in gripper()
        self._holding = True
        self._held_object = obj_name
        return self._make_result("pick", result={
            "holding": True,
            "object": obj_name,
            "grasp_verification": grasp_verification,
        })

    def place(self, x, y, z, hover_height=0.08, steps=80):
        """Place held object at target [x, y, z] position.

        Phase 5:
          - Precondition: must be holding an object.
          - Returns NOT_HOLDING if not holding.
          - Returns target_error after release.
        """
        # Precondition — must be holding
        if not self._holding:
            return self._make_result(
                "place",
                error_code="NOT_HOLDING",
                error_message="Cannot place because the robot is not holding an object.",
            )

        # Phase 2 — validate target via RobotAPI safety check
        # Place target at surface level — validate the hover position as reachable
        ok, err_code, err_msg, _ = self._validate_target(x, y, z + hover_height)
        if not ok:
            return self._make_result("place", error_code=err_code, error_message=err_msg)

        held_obj = getattr(self, "_held_object", "cube") or "cube"
        obj_id = self._sphere_body_id if held_obj == "sphere" and self._sphere_body_id != -1 else self._cube_body_id

        print(f"[robot_api] Placing {held_obj} at ({x:+.3f}, {y:+.3f}, {z:+.3f})...")
        self.move_to(x, y, z + hover_height, steps=steps)
        self.move_to(x, y, z, steps=steps)
        self.gripper(0.0, steps=50)   # Open gripper — release
        self._holding = False          # Authoritative: object released
        self._held_object = None
        self.move_to(x, y, z + hover_height, steps=steps)

        # Measure actual object position after release
        obj_pos = self.data.xpos[obj_id].copy()
        target_pos = np.array([x, y, z], dtype=np.float64)
        target_error = float(np.linalg.norm(obj_pos - target_pos))

        return self._make_result("place", result={
            "released": True,
            "object": held_obj,
            "position": [round(float(obj_pos[0]), 4),
                         round(float(obj_pos[1]), 4),
                         round(float(obj_pos[2]), 4)],
            "target_error": round(target_error, 4),
        })

    def reset_home(self, steps=80):
        """Return arm to home rest position."""
        print("[robot_api] Returning to home position...")
        target_ctrl = HOME_QPOS.copy()
        self._step_to_ctrl(target_ctrl, steps=steps)

        # Phase 1 — reset_home previously returned None; now returns structured result
        eef_pos = self.data.site_xpos[self._site_id].copy()
        return self._make_result("reset_home", result={
            "eef": [round(float(eef_pos[0]), 4),
                    round(float(eef_pos[1]), 4),
                    round(float(eef_pos[2]), 4)],
        })

    def close(self):
        self._running = False
        if self._viewer is not None:
            self._viewer.close()


# ------------------------------------------------------------------ #
# Interactive Terminal Menu & Scenarios
# ------------------------------------------------------------------ #

def run_menu():
    robot = RobotAPI(render=True)
    menu_text = """
========================================
     SO-100 ROBOTIC ARM CONTROL MENU    
========================================
  [1] Info: Telemetry (Arm, Cube, Gripper)
  [2] Grab: Pick up the cube (live tracking)
  [3] Place: Drop cube at target (Right / Left)
  [4] Manual: Move arm to custom (x, y, z)
  [5] Gripper: Open / Close gripper
  [6] Camera: Take snapshot from wrist cam
  --------------------------------------
  SCENARIOS:
  [7] Scenario A: Pick & Place to Right Target
  [8] Scenario B: Pick & Place to Left Target
  [9] Scenario C: Inspection Wave & Camera Photo
  [10] Reset to Home Pose
  [q] Quit Simulation
========================================
"""
    print(menu_text)

    while True:
        try:
            choice = input("Enter choice (1-10 or 'm' for menu, 'q' to quit): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ("q", "quit", "exit"):
            print("Exiting simulation...")
            break

        elif choice in ("m", "menu"):
            print(menu_text)

        elif choice == "1":
            robot.print_info()

        elif choice == "2":
            print("\n>> Running GRAB (Pick)...")
            res = robot.pick()
            print(f">> Pick result: success={res['success']}")
            if res.get("result"):
                print(f"   Grasp verified: {res['result'].get('grasp_verification', {}).get('verified')}")
            robot.print_info()

        elif choice == "3":
            print("\nChoose drop location:")
            print("  1: Right Target Pad (0.15, -0.18, 0.015)")
            print("  2: Left Target Pad  (-0.15, -0.18, 0.015)")
            sub = input("Select (1/2, default 1): ").strip()
            if sub == "2":
                res = robot.place(-0.15, -0.18, 0.015)
            else:
                res = robot.place(0.15, -0.18, 0.015)
            print(f">> Place result: success={res['success']}")

        elif choice == "4":
            print("\nEnter target coordinates in meters (e.g. 0.0 -0.22 0.08):")
            raw = input("X Y Z: ").strip()
            try:
                parts = [float(v) for v in raw.split()]
                if len(parts) != 3:
                    print("Error: Please provide exactly 3 coordinates: X Y Z")
                    continue
                x, y, z = parts
                print(f">> Moving to ({x:.3f}, {y:.3f}, {z:.3f})...")
                res = robot.move_to(x, y, z)
                print(f">> Move result: success={res['success']}")
                if res.get("result"):
                    print(f"   Position error: {res['result'].get('position_error', 'N/A')} m")
            except ValueError:
                print("Invalid numbers entered.")

        elif choice == "5":
            val = input("Set gripper (0 for Open, 1 for Closed, or 0.0 - 1.0): ").strip()
            try:
                g_val = float(val)
                res = robot.gripper(g_val)
                print(f">> Gripper result: success={res['success']}")
            except ValueError:
                print("Invalid value.")

        elif choice == "6":
            filename = input("Filename to save (default: wrist_snapshot.png): ").strip()
            if not filename:
                filename = "wrist_snapshot.png"
            robot.save_camera_image(filename)

        elif choice == "7":
            print("\n=== SCENARIO A: Full Pick & Place to Right ===")
            robot.pick()
            robot.place(0.15, -0.18, 0.015)
            robot.reset_home()
            print("=== Scenario A Finished! ===\n")

        elif choice == "8":
            print("\n=== SCENARIO B: Full Pick & Place to Left ===")
            robot.pick()
            robot.place(-0.15, -0.18, 0.015)
            robot.reset_home()
            print("=== Scenario B Finished! ===\n")

        elif choice == "9":
            print("\n=== SCENARIO C: Inspection Wave & Photo ===")
            print("1. Moving arm up for wide inspection...")
            robot.move_to(0.0, -0.20, 0.16)
            print("2. Snapping high-angle photo...")
            robot.save_camera_image("inspection_high.png")
            print("3. Tilting wrist to inspect left...")
            robot.move_to(-0.12, -0.22, 0.12)
            time.sleep(0.5)
            print("4. Tilting wrist to inspect right...")
            robot.move_to(0.12, -0.22, 0.12)
            time.sleep(0.5)
            print("5. Returning home...")
            robot.reset_home()
            print("=== Scenario C Finished! ===\n")

        elif choice == "10":
            robot.reset_home()

        else:
            print("Unknown command. Type 'm' to see the menu.")

    robot.close()


if __name__ == "__main__":
    run_menu()
