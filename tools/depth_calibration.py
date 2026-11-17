"""Depth calibration for the overhead top_camera.

Renders RGB + metric z-depth from top_camera, detects AprilTag 36h11 markers
with pupil-apriltags, deprojects their centers, and fits a rigid camera->robot
transform (Kabsch/SVD).  Then estimates cube/sphere positions from color masks
(top-face median depth) and compares them to the true MuJoCo body positions.

The top_camera is deliberately mis-placed (offset + tilt) before rendering; the
camera->robot transform is recovered from the tags only.
"""

import os
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from pupil_apriltags import Detector

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "so_arm100",
    "interactive_scene.xml",
)

WIDTH = 1280
HEIGHT = 960
CAMERA_NAME = "top_camera"

# Deliberate camera error applied before rendering (not told to the fit).
CAMERA_OFFSET = np.array([0.015, -0.010, 0.010], dtype=np.float64)
CAMERA_TILT_DEG = 3.0

# Raised pose (arm joints) so the arm does not hide the objects on the table.
RAISED_QPOS = np.array([1.92, -2.5, 0.0, 0.8, 0.0, 1.5], dtype=np.float64)

HALF_HEIGHT = {
    "red_cube": 0.012,
    "green_cube": 0.012,
    "yellow_cube": 0.012,
    "blue_sphere": 0.014,
}

TOP_FACE_BAND_M = 0.006
NOISE_SIGMA_M = 0.002
NOISE_TRIALS = 20


def load_model():
    model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
    # Offscreen framebuffer must be large enough for 1280x960.
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    return model


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


def perturb_camera(model):
    """Offset and tilt top_camera in the model (worldbody camera)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    model.cam_pos[cam_id] += CAMERA_OFFSET
    theta = np.deg2rad(CAMERA_TILT_DEG)
    q_delta = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0])
    q_new = quat_mul(q_delta, model.cam_quat[cam_id].copy())
    model.cam_quat[cam_id] = q_new / np.linalg.norm(q_new)
    return cam_id


def build_intrinsics(model):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    fovy = float(model.cam_fovy[cam_id])
    fy = (HEIGHT / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy
    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0
    return fx, fy, cx, cy, fovy


def read_tag_positions():
    """Read the 4 apriltag geom centers (x, y, z) from the XML."""
    root = ET.parse(_MODEL_PATH).getroot()
    tags = {}
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name.startswith("apriltag_"):
            tag_id = int(name.rsplit("_", 1)[1])
            tags[tag_id] = np.array([float(v) for v in geom.get("pos").split()])
    return tags


def deproject(u, v, z, fx, fy, cx, cy):
    """Pixel + metric z-depth -> 3D point in the OpenCV camera frame."""
    return np.array(
        [(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64
    )


def kabsch(cam_points, robot_points):
    """Rigid transform (R, t) mapping camera points to robot points, no scaling."""
    P = np.asarray(cam_points, dtype=np.float64)
    Q = np.asarray(robot_points, dtype=np.float64)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def color_masks(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    base = (s > 90) & (v > 60)
    return {
        "red_cube": (base & ((h <= 8) | (h >= 172))).astype(np.uint8),
        "green_cube": (base & (h >= 55) & (h <= 82)).astype(np.uint8),
        "yellow_cube": (base & (h >= 18) & (h <= 42)).astype(np.uint8),
        "blue_sphere": (base & (h >= 88) & (h <= 110)).astype(np.uint8),
    }


def object_component(mask):
    """Largest plausible object blob (excludes the larger semi-transparent pads)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best_i = None
    best_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 300 <= area <= 3000 and area > best_area:
            best_area = area
            best_i = i
    if best_i is None:
        return None
    return labels == best_i


def render(model, data):
    rgb_renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    depth_renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    depth_renderer.enable_depth_rendering()

    rgb_renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = rgb_renderer.render()

    depth_renderer.update_scene(data, camera=CAMERA_NAME)
    depth = depth_renderer.render()

    rgb_renderer.close()
    depth_renderer.close()
    return rgb, depth


def top_face_estimate(comp_mask, depth_image, fx, fy, cx, cy, R, t):
    """Median depth of the top-face band + mean pixel coords -> robot-frame xyz."""
    mask_u8 = comp_mask.astype(np.uint8) * 255
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=2) > 0
    ys, xs = np.nonzero(eroded)
    d = depth_image[ys, xs]
    valid = d > 0
    ys, xs, d = ys[valid], xs[valid], d[valid]
    if d.size == 0:
        return None
    ref = float(np.percentile(d, 20))
    keep = d <= ref + TOP_FACE_BAND_M
    u = float(xs[keep].mean())
    v = float(ys[keep].mean())
    z = float(np.median(d[keep]))
    cam_pt = deproject(u, v, z, fx, fy, cx, cy)
    est = R @ cam_pt + t
    return est, u, v, z


def print_object_table(title, rows):
    print(f"\n{title}")
    print(
        f"{'object':<13}{'est xyz (m)':<28}{'true xyz (m)':<28}"
        f"{'xy(mm)':>8}{'raw z(mm)':>11}{'corr z(mm)':>12}"
        f"{'raw 3D(mm)':>12}{'corr 3D(mm)':>13}"
    )
    print("-" * 125)
    for name, est, true, xy, raw_z, corr_z, raw_3d, corr_3d in rows:
        print(
            f"{name:<13}"
            f"({est[0]:+.4f}, {est[1]:+.4f}, {est[2]:+.4f})   "
            f"({true[0]:+.4f}, {true[1]:+.4f}, {true[2]:+.4f})   "
            f"{xy:8.2f}{raw_z:11.2f}{corr_z:12.2f}{raw_3d:12.2f}{corr_3d:13.2f}"
        )
    if rows:
        print(
            f"{'MEAN':<13}{'':<56}"
            f"{np.mean([r[3] for r in rows]):8.2f}"
            f"{np.mean([r[4] for r in rows]):11.2f}"
            f"{np.mean([r[5] for r in rows]):12.2f}"
            f"{np.mean([r[6] for r in rows]):12.2f}"
            f"{np.mean([r[7] for r in rows]):13.2f}"
        )


def evaluate_objects(components, depth_image, true_positions, fx, fy, cx, cy, R, t):
    rows = []
    for name, comp in components.items():
        if comp is None:
            print(f"  WARNING: could not locate {name}")
            continue
        result = top_face_estimate(comp, depth_image, fx, fy, cx, cy, R, t)
        if result is None:
            continue
        est, _, _, _ = result
        true = true_positions[name]
        est_corr = est.copy()
        est_corr[2] -= HALF_HEIGHT[name]
        xy = float(np.linalg.norm(est[:2] - true[:2]) * 1000.0)
        raw_z = float(abs(est[2] - true[2]) * 1000.0)
        corr_z = float(abs(est_corr[2] - true[2]) * 1000.0)
        raw_3d = float(np.linalg.norm(est - true) * 1000.0)
        corr_3d = float(np.linalg.norm(est_corr - true) * 1000.0)
        rows.append((name, est, true, xy, raw_z, corr_z, raw_3d, corr_3d))
    return rows


def main():
    model = load_model()
    data = mujoco.MjData(model)

    perturb_camera(model)
    fx, fy, cx, cy, fovy = build_intrinsics(model)
    print(f"Camera perturbation: dpos={CAMERA_OFFSET.tolist()} m, tilt={CAMERA_TILT_DEG} deg about x")
    print(f"Intrinsics: fx={fx:.3f} fy={fy:.3f} cx={cx:.1f} cy={cy:.1f} (fovy={fovy:.1f} deg)")

    # Raised arm pose, set directly (no viewer, no physics stepping).
    data.qpos[:6] = RAISED_QPOS
    data.ctrl[:6] = RAISED_QPOS
    mujoco.mj_forward(model, data)

    rgb, depth = render(model, data)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # ---- Detect tags and deproject their centers ---------------------------
    detector = Detector(families="tag36h11")
    detections = detector.detect(gray)
    detected = {int(d.tag_id): (float(d.center[0]), float(d.center[1])) for d in detections}
    print(f"Detected tag ids: {sorted(detected)}")

    true_tags = read_tag_positions()
    cam_points = []
    robot_points = []
    tag_ids = []
    for tag_id in sorted(detected):
        u, v = detected[tag_id]
        z = float(depth[int(round(v)), int(round(u))])
        cam_points.append(deproject(u, v, z, fx, fy, cx, cy))
        robot_points.append(true_tags[tag_id])
        tag_ids.append(tag_id)
        print(f"  tag {tag_id}: pixel=({u:.2f}, {v:.2f}) depth={z:.4f} m")

    # ---- Rigid transform (Kabsch), recovered from tags only ----------------
    R, t = kabsch(cam_points, robot_points)
    print("\nT (camera -> robot):")
    print("R =")
    for row in R:
        print(f"  [{row[0]: .6f} {row[1]: .6f} {row[2]: .6f}]")
    print(f"t = [{t[0]: .6f} {t[1]: .6f} {t[2]: .6f}]")

    print("\nTag residuals (mm):")
    for tag_id, pc, pr in zip(tag_ids, cam_points, robot_points):
        residual = np.linalg.norm(R @ pc + t - pr) * 1000.0
        print(f"  tag {tag_id}: {residual:.3f}")

    # ---- Object estimates (median top-face depth) --------------------------
    masks = color_masks(rgb)
    components = {name: object_component(mask) for name, mask in masks.items()}
    true_positions = {
        name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
        for name in HALF_HEIGHT
    }

    clean_rows = evaluate_objects(
        components, depth, true_positions, fx, fy, cx, cy, R, t
    )
    print_object_table("Object estimates (clean depth, median top-face method)", clean_rows)

    # ---- Noise run: 20 trials, sigma = 2 mm --------------------------------
    rng = np.random.default_rng(0)
    per_object = {name: {"xy": [], "corr_z": [], "corr_3d": []} for name in HALF_HEIGHT}
    for _ in range(NOISE_TRIALS):
        noisy_depth = depth + rng.normal(0.0, NOISE_SIGMA_M, size=depth.shape)
        rows = evaluate_objects(
            components, noisy_depth, true_positions, fx, fy, cx, cy, R, t
        )
        for name, _, _, xy, _, corr_z, _, corr_3d in rows:
            per_object[name]["xy"].append(xy)
            per_object[name]["corr_z"].append(corr_z)
            per_object[name]["corr_3d"].append(corr_3d)

    print(
        f"\nNoise run (sigma={NOISE_SIGMA_M * 1000:.0f} mm, {NOISE_TRIALS} trials, "
        "median top-face method)"
    )
    print(
        f"{'object':<13}{'mean xy':>10}{'max xy':>9}"
        f"{'mean corr z':>13}{'max corr z':>12}"
        f"{'mean corr 3D':>15}{'max corr 3D':>14}"
    )
    print("-" * 86)
    for name in HALF_HEIGHT:
        xy = np.array(per_object[name]["xy"])
        cz = np.array(per_object[name]["corr_z"])
        c3 = np.array(per_object[name]["corr_3d"])
        if xy.size == 0:
            continue
        print(
            f"{name:<13}{xy.mean():10.2f}{xy.max():9.2f}"
            f"{cz.mean():13.2f}{cz.max():12.2f}"
            f"{c3.mean():15.2f}{c3.max():14.2f}"
        )


if __name__ == "__main__":
    main()
