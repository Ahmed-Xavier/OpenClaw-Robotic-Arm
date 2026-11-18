"""Army Perception Studio - snapshot exporter.

Runs the simulation headless, makes Army look at the table (same observe()
call the agent uses), takes ONE camera frame, and writes a single
self-contained file: studio/army_studio.html  (open it in any browser).

Nothing in the existing project files is changed.

Usage (from the repo root):
    python studio/export_scene.py
    python studio/export_scene.py --layout 2        # one of the 5 test layouts
    python studio/export_scene.py --noise 0.002     # add 2 mm depth noise
"""

import argparse
import base64
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from perception import (  # noqa: E402
    HALF_HEIGHT,
    TAG_WORLD,
    TopCameraPerception,
    _MODEL_PATH,
    _color_mask,
    deproject,
)
from robot_api import RobotAPI  # noqa: E402
from sim_camera import SimTopCamera  # noqa: E402
from vision_state import observe  # noqa: E402
from test_observe import (  # noqa: E402
    OBJECTS,
    generate_layouts,
    object_qpos_addresses,
    set_layout,
)

LOOK_POSE_PATH = os.path.join(ROOT, "tools", "look_pose.json")
TEMPLATE_PATH = os.path.join(HERE, "viewer_template.html")
OUT_PATH = os.path.join(HERE, "army_studio.html")

# Raw cloud: keep only points inside this box (robot frame, metres).
CLOUD_BOX = dict(x=(-0.60, 0.60), y=(-0.70, 0.25), z=(-0.02, 0.60))


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def read_tag_sizes():
    """Half-size (x, y) of each AprilTag square, from the scene XML."""
    sizes = {}
    for geom in ET.parse(_MODEL_PATH).getroot().iter("geom"):
        name = geom.get("name", "")
        if name.startswith("apriltag_"):
            tag_id = int(name.rsplit("_", 1)[1])
            sizes[tag_id] = [float(v) for v in geom.get("size").split()][:2]
    return sizes


def robot_frame_grids(depth, K, T):
    """Per-pixel X, Y, Z in the robot frame (NaN where depth is invalid)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = np.isfinite(depth) & (depth > 0)
    pts = deproject(u[valid], v[valid], depth[valid], K) @ T[:3, :3].T + T[:3, 3]
    out = np.full((3, h, w), np.nan)
    for i in range(3):
        out[i][valid] = pts[:, i]
    return out[0], out[1], out[2], valid


def footprint(rgb, X, Y, Z, valid, name):
    """Top-view silhouette of one object as a polygon (metres, robot frame)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sel = _color_mask(hsv, name) & valid & (Z > 0.008)
    if np.count_nonzero(sel) < 30:
        return None
    pts_mm = np.stack([X[sel], Y[sel]], axis=1).astype(np.float32) * 1000.0
    hull = cv2.convexHull(pts_mm)
    hull = cv2.approxPolyDP(hull, 0.7, True).reshape(-1, 2)
    return (hull / 1000.0).tolist()


def view_corners_on_table(K, T, w, h):
    """Where the 4 image corners land on the table plane (z = 0)."""
    corners = []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        d_cam = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
        d = T[:3, :3] @ d_cam
        s = -T[2, 3] / d[2]
        p = T[:3, 3] + s * d
        corners.append([float(p[0]), float(p[1]), 0.0])
    return corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=int, default=None,
                    help="0-4: use one of the random test layouts (default: scene as is)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="depth noise sigma in metres (default 0)")
    ap.add_argument("--stride", type=int, default=3,
                    help="keep 1 pixel out of N in the raw cloud (default 3)")
    args = ap.parse_args()

    with open(LOOK_POSE_PATH, "r", encoding="utf-8") as fh:
        look_qpos = np.asarray(json.load(fh)["qpos"], dtype=np.float64)

    robot = RobotAPI(render=False)
    cam = SimTopCamera(robot.model, robot.data, depth_noise_sigma=args.noise)
    perception = TopCameraPerception()
    try:
        if args.layout is not None:
            addresses = object_qpos_addresses(robot.model)
            layout = generate_layouts(seed=11)[args.layout]
            set_layout(robot.model, robot.data, addresses, layout)

        # 1) Same call the agent uses: moves to the look pose, checks calibration.
        result = observe(robot, cam, perception, look_qpos)
        if not result.get("success"):
            sys.exit(f"observe failed: {result.get('error')}")

        # 2) One frame, used for EVERYTHING below so the views agree.
        rgb, depth, K = cam.grab()
        T, residuals_mm = perception.calibrate(rgb, depth, K)
        objects = perception.detect_objects(rgb, depth, K, T)
        X, Y, Z, valid = robot_frame_grids(depth, K, T)

        # Sim truth (only exists in simulation).
        truth = {}
        for name in OBJECTS:
            bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            truth[name] = robot.data.xpos[bid].astype(float).tolist()

        obj_out = []
        for name in OBJECTS:
            info = objects[name]
            obj_out.append({
                "name": name,
                "visible": bool(info["visible"]),
                "xyz": info["xyz"],
                "top_z": info["top_z"],
                "n_pixels": info["n_pixels"],
                "half_height_prior": HALF_HEIGHT[name],
                "footprint": footprint(rgb, X, Y, Z, valid, name) if info["visible"] else None,
                "truth_xyz": truth[name],
            })

        # Raw cloud (down-sampled).
        s = max(1, args.stride)
        sl = (slice(None, None, s), slice(None, None, s))
        Xs, Ys, Zs, Vs, C = X[sl], Y[sl], Z[sl], valid[sl], rgb[sl]
        keep = (
            Vs
            & (Xs > CLOUD_BOX["x"][0]) & (Xs < CLOUD_BOX["x"][1])
            & (Ys > CLOUD_BOX["y"][0]) & (Ys < CLOUD_BOX["y"][1])
            & (Zs > CLOUD_BOX["z"][0]) & (Zs < CLOUD_BOX["z"][1])
        )
        pos = np.stack([Xs[keep], Ys[keep], Zs[keep]], axis=1).astype(np.float32)
        col = C[keep].astype(np.uint8)

        h, w = depth.shape
        sizes = read_tag_sizes()
        snapshot = {
            "meta": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "units": "m (robot frame, z up)",
                "image": [w, h],
                "layout": args.layout,
                "depth_noise_sigma_m": args.noise,
                "cloud_stride": s,
                "n_cloud_points": int(pos.shape[0]),
                "calibration_residual_mm_max": float(max(residuals_mm.values())),
                "source": result["source"],
            },
            "camera": {
                "position": T[:3, 3].tolist(),
                "right": T[:3, 0].tolist(),
                "down": T[:3, 1].tolist(),
                "forward": T[:3, 2].tolist(),
                "view_corners_on_table": view_corners_on_table(K, T, w, h),
            },
            "tags": [
                {"id": i, "xyz": list(TAG_WORLD[i]), "half_size": sizes[i],
                 "residual_mm": residuals_mm.get(i)}
                for i in sorted(TAG_WORLD)
            ],
            "objects": obj_out,
            "cloud": {"positions_f32_b64": b64(pos), "colors_u8_b64": b64(col)},
        }
    finally:
        cam.close()
        robot.close()

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace("__SNAPSHOT_JSON__", json.dumps(snapshot))
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"wrote {OUT_PATH}  ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")
    print(f"cloud points: {pos.shape[0]}   max tag residual: "
          f"{snapshot['meta']['calibration_residual_mm_max']:.2f} mm")
    for o in obj_out:
        state = "ok " if o["visible"] else "MISSING"
        print(f"  {o['name']:<12} {state}", "" if not o["visible"] else
              f"xyz(mm)=({o['xyz'][0]*1000:.1f}, {o['xyz'][1]*1000:.1f}, {o['xyz'][2]*1000:.1f})")


if __name__ == "__main__":
    main()
