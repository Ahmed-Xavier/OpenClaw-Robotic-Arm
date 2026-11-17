"""Evaluate top-camera perception over 30 random object layouts and 3 conditions."""

import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perception import HALF_HEIGHT, TAG_WORLD, TopCameraPerception  # noqa: E402
from sim_camera import SimTopCamera  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "third_party", "so_arm100", "interactive_scene.xml")
LOOK_POSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "look_pose.json")

OBJECTS = ("red_cube", "green_cube", "yellow_cube", "blue_sphere")
TARGET_PADS = ((0.15, -0.18), (-0.15, -0.18))
N_LAYOUTS = 30
NOISE_SIGMA = 0.002
CAMERA_OFFSET = np.array([0.015, -0.010, 0.010], dtype=np.float64)
CAMERA_TILT_DEG = 3.0
WIDTH, HEIGHT = 1280, 960


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def object_qpos_addresses(model):
    addresses = {}
    for name in OBJECTS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = model.body_jntadr[body_id]
        addresses[name] = model.jnt_qposadr[joint_id]
    return addresses


def generate_layouts(seed=0):
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


def set_layout(model, data, addresses, layout):
    for name, xy in layout.items():
        adr = addresses[name]
        z = float(HALF_HEIGHT[name])
        data.qpos[adr : adr + 3] = [float(xy[0]), float(xy[1]), z]
        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def perturb_camera(model):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera")
    model.cam_pos[cam_id] += CAMERA_OFFSET
    theta = np.deg2rad(CAMERA_TILT_DEG)
    q_delta = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0])
    q_new = quat_mul(q_delta, model.cam_quat[cam_id].copy())
    model.cam_quat[cam_id] = q_new / np.linalg.norm(q_new)


def run_condition(model, data, addresses, layouts, perception, look_qpos, sigma):
    cam = SimTopCamera(
        model,
        data,
        width=WIDTH,
        height=HEIGHT,
        depth_noise_sigma=sigma,
    )
    stats = {name: {"detected": 0, "xy": [], "z": [], "d3": []} for name in OBJECTS}
    try:
        for layout in layouts:
            set_layout(model, data, addresses, layout)
            data.qpos[:6] = look_qpos
            data.ctrl[:6] = look_qpos
            mujoco.mj_forward(model, data)

            true_positions = {
                name: data.xpos[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                ].copy()
                for name in OBJECTS
            }

            rgb, depth, K = cam.grab()
            T, _ = perception.calibrate(rgb, depth, K)
            detected = perception.detect_objects(rgb, depth, K, T)

            for name in OBJECTS:
                info = detected[name]
                if not info["visible"] or info["xyz"] is None:
                    continue
                est = np.asarray(info["xyz"], dtype=np.float64)
                true = true_positions[name]
                stats[name]["detected"] += 1
                stats[name]["xy"].append(float(np.linalg.norm(est[:2] - true[:2]) * 1000.0))
                stats[name]["z"].append(float(abs(est[2] - true[2]) * 1000.0))
                stats[name]["d3"].append(float(np.linalg.norm(est - true) * 1000.0))
    finally:
        cam.close()
    return stats


def print_condition(title, stats):
    print(f"\n{title}")
    print(
        f"{'object':<14}{'detect':>8}{'mean xy':>10}{'max xy':>9}"
        f"{'mean corr z':>13}{'max corr z':>12}"
        f"{'mean 3D':>10}{'max 3D':>9}"
    )
    print("-" * 85)
    for name in OBJECTS:
        s = stats[name]
        if s["detected"] == 0:
            print(f"{name:<14}{0.0:7.1f}%{'':>9}{'':>9}{'':>13}{'':>12}{'':>10}{'':>9}")
            continue
        xy = np.asarray(s["xy"])
        z = np.asarray(s["z"])
        d3 = np.asarray(s["d3"])
        print(
            f"{name:<14}{100.0 * s['detected'] / N_LAYOUTS:7.1f}%"
            f"{xy.mean():10.2f}{xy.max():9.2f}"
            f"{z.mean():13.2f}{z.max():12.2f}"
            f"{d3.mean():10.2f}{d3.max():9.2f}"
        )


def condition_passes(stats, clean):
    for name in OBJECTS:
        s = stats[name]
        if s["detected"] != N_LAYOUTS:
            return False
        xy_mean = float(np.mean(s["xy"]))
        d3_max = float(np.max(s["d3"]))
        if clean:
            if xy_mean >= 1.0 or d3_max >= 3.0:
                return False
        else:
            if xy_mean >= 1.5 or d3_max >= 4.0:
                return False
    sphere_z_mean = float(np.mean(stats["blue_sphere"]["z"]))
    return sphere_z_mean < 1.5


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    if model.vis.global_.offwidth < WIDTH:
        model.vis.global_.offwidth = WIDTH
    if model.vis.global_.offheight < HEIGHT:
        model.vis.global_.offheight = HEIGHT

    with open(LOOK_POSE_PATH, "r", encoding="utf-8") as fh:
        look_qpos = np.asarray(json.load(fh)["qpos"], dtype=np.float64)

    addresses = object_qpos_addresses(model)
    layouts = generate_layouts(seed=0)
    perception = TopCameraPerception()

    data.qpos[:6] = look_qpos
    data.ctrl[:6] = look_qpos
    mujoco.mj_forward(model, data)

    print(f"layouts: {N_LAYOUTS}")
    print(f"look qpos: {[round(float(v), 6) for v in look_qpos]}")

    clean_stats = run_condition(
        model, data, addresses, layouts, perception, look_qpos, 0.0
    )
    print_condition("condition (a): clean", clean_stats)

    noisy_stats = run_condition(
        model, data, addresses, layouts, perception, look_qpos, NOISE_SIGMA
    )
    print_condition(
        f"condition (b): depth noise sigma {NOISE_SIGMA * 1000:.0f} mm", noisy_stats
    )

    perturb_camera(model)
    print(
        f"\ncamera perturbation (c): offset={CAMERA_OFFSET.tolist()} m, "
        f"tilt={CAMERA_TILT_DEG} deg about x"
    )
    perturbed_stats = run_condition(
        model, data, addresses, layouts, perception, look_qpos, NOISE_SIGMA
    )
    print_condition(
        f"condition (c): camera offset/tilt, depth noise sigma {NOISE_SIGMA * 1000:.0f} mm",
        perturbed_stats,
    )

    results = {
        "a_clean": condition_passes(clean_stats, clean=True),
        "b_noise": condition_passes(noisy_stats, clean=False),
        "c_camera": condition_passes(perturbed_stats, clean=False),
    }
    print("\nPASS/FAIL")
    print(f"  (a) clean:                         {'PASS' if results['a_clean'] else 'FAIL'}")
    print(f"  (b) depth noise 2 mm:              {'PASS' if results['b_noise'] else 'FAIL'}")
    print(f"  (c) camera offset/tilt + noise:    {'PASS' if results['c_camera'] else 'FAIL'}")
    overall = all(results.values())
    print(f"  overall:                           {'PASS' if overall else 'FAIL'}")
    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
