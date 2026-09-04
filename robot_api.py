"""
robot_api.py — Robot Abstraction Layer for the MuJoCo-simulated SO-ARM100 (SO-100).

Hardware-independent API for the Hugging Face LeRobot SO-100 arm:
    move_to(x, y, z, steps=80)
    pick(hover_height=0.08)
    place(x, y, z, hover_height=0.08)
    gripper(value)               # 0.0 = fully open, 1.0 = fully closed
    get_state()                  # Full snapshot of arm, gripper, cube, and joint states
    capture_image()              # Returns RGB array from gripper camera
    save_camera_image()          # Saves camera image to disk
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
HOME_QPOS = np.array([0.0, -1.38, 1.79, 1.34, 0.0, JAW_OPEN], dtype=np.float64)

JOINT_NAMES = ["Rotation (Base)", "Pitch (Shoulder)", "Elbow", "Wrist Pitch", "Wrist Roll", "Jaw (Gripper)"]




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

        return best_q

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
    # Public Robot API
    # ------------------------------------------------------------------ #

    def get_state(self):
        """Return a structured dictionary of live simulation telemetry."""
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        cube_vel = self.data.cvel[self._cube_body_id].copy()
        eef_pos = self.data.site_xpos[self._site_id].copy()
        joint_qpos = self.data.qpos[:6].tolist()
        jaw_pos = float(self.data.qpos[5])

        dist_to_cube = float(np.linalg.norm(eef_pos - cube_pos))

        return {
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

    def move_to(self, x, y, z, steps=80):
        """Move the end-effector (grasp site) to an absolute [x, y, z] target."""
        target_pos = np.array([x, y, z], dtype=np.float64)
        target_q5 = self._solve_ik(target_pos, q_seed=self.data.ctrl[:5])

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[:5] = target_q5
        self._step_to_ctrl(target_ctrl, steps=steps)
        return self.get_state()

    def gripper(self, value, steps=60):
        """Set gripper openness: 0.0 = fully open, 1.0 = fully closed."""
        val = float(np.clip(value, 0.0, 1.0))
        jaw_target = (1.0 - val) * JAW_OPEN + val * JAW_CLOSED

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[5] = jaw_target
        self._step_to_ctrl(target_ctrl, steps=steps)
        self._holding = val > 0.5
        return self.get_state()

    def pick(self, approach_dist=0.12, hover_height=0.08, steps=80):
        """Pick up the cube with a horizontal side approach.

        Instead of hovering above and descending (which drives the jaw tip
        into the top of the object), the arm:
          1. Opens the gripper
          2. Moves to cube height at a standoff distance behind the cube
             (arm-base → cube direction, shifted back by approach_dist)
          3. Slides in horizontally to the cube — jaw slides around it, not onto it
          4. Settles, then closes the gripper
          5. Dwells 0.5 s to establish the grip
          6. Lifts straight up
        """
        if self._holding:
            print("[robot_api] Already holding the cube — skipping grab.")
            return self.get_state()

        cube_pos = self.data.xpos[self._cube_body_id].copy()
        x, y, z = cube_pos[0], cube_pos[1], cube_pos[2]

        # Direction vector from arm base (origin) toward the cube, XY only
        vec = np.array([x, y, 0.0])
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            approach_dir = np.array([0.0, -1.0, 0.0])
        else:
            approach_dir = vec / dist   # unit vector toward cube

        # Standoff position: cube height, approach_dist further away from the arm
        sx = x - approach_dir[0] * approach_dist
        sy = y - approach_dir[1] * approach_dist

        print(f"[robot_api] Cube at ({x:+.3f}, {y:+.3f}, {z:+.3f})")
        print(f"[robot_api] Side-approach standoff: ({sx:+.3f}, {sy:+.3f}, {z:+.3f})")

        # 1. Open gripper
        self.gripper(0.0, steps=40)

        # 2. Move to standoff at cube height (no hover — stay at cube's Z)
        print("[robot_api] Moving to side standoff...")
        self.move_to(sx, sy, z, steps=steps)

        # 3. Slide in horizontally — jaw comes in from the side, not from above
        print("[robot_api] Sliding in horizontally to cube...")
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
        return self.get_state()

    def place(self, x, y, z, hover_height=0.08, steps=80):
        """Place held object at target [x, y, z] position."""
        print(f"[robot_api] Placing at ({x:+.3f}, {y:+.3f}, {z:+.3f})...")
        self.move_to(x, y, z + hover_height, steps=steps)
        self.move_to(x, y, z, steps=steps)
        self.gripper(0.0, steps=50)  # Open gripper
        self.move_to(x, y, z + hover_height, steps=steps)
        return self.get_state()

    def reset_home(self, steps=80):
        """Return arm to home rest position."""
        print("[robot_api] Returning to home position...")
        target_ctrl = HOME_QPOS.copy()
        self._step_to_ctrl(target_ctrl, steps=steps)

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
            robot.pick()
            print(">> Pick complete! State:")
            robot.print_info()

        elif choice == "3":
            print("\nChoose drop location:")
            print("  1: Right Target Pad (0.15, -0.18, 0.015)")
            print("  2: Left Target Pad  (-0.15, -0.18, 0.015)")
            sub = input("Select (1/2, default 1): ").strip()
            if sub == "2":
                robot.place(-0.15, -0.18, 0.015)
            else:
                robot.place(0.15, -0.18, 0.015)
            print(">> Place complete!")

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
                robot.move_to(x, y, z)
                print(">> Movement complete!")
            except ValueError:
                print("Invalid numbers entered.")

        elif choice == "5":
            val = input("Set gripper (0 for Open, 1 for Closed, or 0.0 - 1.0): ").strip()
            try:
                g_val = float(val)
                robot.gripper(g_val)
                print(f">> Gripper set to {g_val}!")
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
