"""End-to-end check of vision_state.observe() on a real RobotAPI (no viewer)."""

import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perception import HALF_HEIGHT, TAG_WORLD, TopCameraPerception  # noqa: E402
from robot_api import RobotAPI  # noqa: E402
from sim_camera import SimTopCamera  # noqa: E402
from vision_state import observe  # noqa: E402

LOOK_POSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "look_pose.json")
OBJECTS = ("red_cube", "green_cube", "yellow_cube", "blue_sphere")
TARGET_PADS = ((0.15, -0.18), (-0.15, -0.18))
N_LAYOUTS = 5
LOOK_POSE_TOL_RAD = 0.03


def generate_layouts(seed=11):
    rng = np.random.default_rng(seed)
    tags = [np.array(TAG_WORLD[i][:2]) for i in range(4)]
    obstacles = tags + [np.array(p) for p in TARGET_PADS]
    layouts = []
    for _ in range(N_LAYOUTS):
        chosen = []
        for _name in OBJECTS:
            while True:
                p = np.array(
                    [rng.uniform(-0.25, 0.25), rng.uniform(-0.32, -0.12)]
                )
                if any(np.linalg.norm(p - q) < 0.05 for q in chosen):
                    continue
                if any(np.linalg.norm(p - q) < 0.04 for q in obstacles):
                    continue
                chosen.append(p)
                break
        layouts.append({name: p for name, p in zip(OBJECTS, chosen)})
    return layouts


def object_qpos_addresses(model):
    addresses = {}
    for name in OBJECTS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = model.body_jntadr[body_id]
        addresses[name] = model.jnt_qposadr[joint_id]
    return addresses


def set_layout(model, data, addresses, layout):
    for name, xy in layout.items():
        adr = addresses[name]
        z = float(HALF_HEIGHT[name])
        data.qpos[adr : adr + 3] = [float(xy[0]), float(xy[1]), z]
        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def main():
    with open(LOOK_POSE_PATH, "r", encoding="utf-8") as fh:
        look_qpos = np.asarray(json.load(fh)["qpos"], dtype=np.float64)

    robot = RobotAPI(render=False)
    addresses = object_qpos_addresses(robot.model)
    layouts = generate_layouts(seed=11)
    perception = TopCameraPerception()
    cam = SimTopCamera(robot.model, robot.data)

    stats = {name: {"detected": 0, "xy": [], "z": [], "d3": []} for name in OBJECTS}
    residual_max = 0.0
    look_errors = []

    try:
        for layout in layouts:
            set_layout(robot.model, robot.data, addresses, layout)

            result = observe(robot, cam, perception, look_qpos)
            if not result.get("success"):
                print(f"observe failed: {result.get('error')}")
                sys.exit(1)

            residual_max = max(
                residual_max, float(result["calibration_residual_mm_max"])
            )
            look_errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(robot.data.qpos[:5], dtype=np.float64)
                            - look_qpos[:5]
                        )
                    )
                )
            )

            true_positions = {
                name: robot.data.xpos[
                    mujoco.mj_name2id(
                        robot.model, mujoco.mjtObj.mjOBJ_BODY, name
                    )
                ].copy()
                for name in OBJECTS
            }

            for name in OBJECTS:
                info = result["objects"][name]
                if not info["visible"] or info["xyz"] is None:
                    continue
                est = np.asarray(info["xyz"], dtype=np.float64)
                true = true_positions[name]
                stats[name]["detected"] += 1
                stats[name]["xy"].append(
                    float(np.linalg.norm(est[:2] - true[:2]) * 1000.0)
                )
                stats[name]["z"].append(float(abs(est[2] - true[2]) * 1000.0))
                stats[name]["d3"].append(
                    float(np.linalg.norm(est - true) * 1000.0)
                )
    finally:
        cam.close()
        robot.close()

    print(f"\nlayouts: {N_LAYOUTS}")
    print(f"max calibration residual: {residual_max:.3f} mm")
    print(
        f"{'object':<14}{'detect':>9}{'mean xy':>10}{'max xy':>9}"
        f"{'mean corr z':>13}{'max corr z':>12}"
        f"{'mean 3D':>10}{'max 3D':>9}"
    )
    print("-" * 86)
    for name in OBJECTS:
        s = stats[name]
        if s["detected"] == 0:
            print(f"{name:<14}{0.0:8.1f}%")
            continue
        xy = np.asarray(s["xy"])
        z = np.asarray(s["z"])
        d3 = np.asarray(s["d3"])
        print(
            f"{name:<14}{100.0 * s['detected'] / N_LAYOUTS:8.1f}%"
            f"{xy.mean():10.2f}{xy.max():9.2f}"
            f"{z.mean():13.2f}{z.max():12.2f}"
            f"{d3.mean():10.2f}{d3.max():9.2f}"
        )

    worst_look_error = max(look_errors)
    reached = all(err < LOOK_POSE_TOL_RAD for err in look_errors)
    print(
        f"\nlook pose reached: {'YES' if reached else 'NO'} "
        f"(max |qpos[:5]-look|={worst_look_error:.4f} rad, tol={LOOK_POSE_TOL_RAD:.2f})"
    )
    detection_ok = all(stats[name]["detected"] == N_LAYOUTS for name in OBJECTS)
    print(f"detection: {'PASS' if detection_ok else 'FAIL'}")
    if not (reached and detection_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
