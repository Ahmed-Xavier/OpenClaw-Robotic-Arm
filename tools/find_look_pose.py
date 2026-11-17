"""Search for a compact "Z" look pose that keeps the arm out of the object zone.

The look pose is the arm folded back so that no arm pixel falls inside the
object zone, while all four AprilTags stay visible and no arm geom touches the
table.
"""

import json
import os
import sys

import cv2
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perception import Intrinsics, TAG_WORLD  # noqa: E402
from robot_api import HOME_QPOS  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "third_party", "so_arm100", "interactive_scene.xml")
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
MANUAL_PATH = os.path.join(TOOLS_DIR, "look_pose_manual.json")
LOOK_POSE_PATH = os.path.join(TOOLS_DIR, "look_pose.json")
TOP_VIEW_PATH = os.path.join(TOOLS_DIR, "look_pose_top.png")

WIDTH, HEIGHT = 640, 480
DILATE_PX = 15
ZONE_X = (-0.30, 0.30)
ZONE_Y = (-0.36, -0.08)
TAG_VISIBLE_MIN = 0.80
GRID = (5, 6, 6, 5)


def build_intrinsics(model, camera_id, width, height):
    fovy = float(model.cam_fovy[camera_id])
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    return Intrinsics(fx=fy, fy=fy, cx=width / 2.0, cy=height / 2.0)


def world_to_pixel(points, data, camera_id, K):
    R = data.cam_xmat[camera_id].reshape(3, 3)
    cpos = data.cam_xpos[camera_id]
    local = (np.asarray(points, dtype=np.float64) - cpos) @ R
    x = local[:, 0]
    y = -local[:, 1]
    z = -local[:, 2]
    u = K.fx * x / z + K.cx
    v = K.fy * y / z + K.cy
    return np.stack([u, v], axis=1)


def build_zone_mask(data, camera_id, K):
    corners = np.array(
        [
            [ZONE_X[0], ZONE_Y[0], 0.0],
            [ZONE_X[1], ZONE_Y[0], 0.0],
            [ZONE_X[1], ZONE_Y[1], 0.0],
            [ZONE_X[0], ZONE_Y[1], 0.0],
        ]
    )
    uv = world_to_pixel(corners, data, camera_id, K)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(uv).astype(np.int32), 1)
    kernel = np.ones((2 * DILATE_PX + 1, 2 * DILATE_PX + 1), dtype=np.uint8)
    return cv2.dilate(mask, kernel)


def build_tag_polygons(data, camera_id, K, tag_geoms):
    polygons = {}
    for tag_id, geom_id in tag_geoms.items():
        gx, gy, gz = TAG_WORLD[tag_id]
        s = 0.02
        corners = np.array(
            [
                [gx - s, gy - s, gz],
                [gx + s, gy - s, gz],
                [gx + s, gy + s, gz],
                [gx - s, gy + s, gz],
            ]
        )
        uv = world_to_pixel(corners, data, camera_id, K)
        poly = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.fillConvexPoly(poly, np.round(uv).astype(np.int32), 1)
        expected_area = max(float(np.count_nonzero(poly)), 1.0)
        polygons[tag_id] = (poly, expected_area)
    return polygons


def arm_descendants(model):
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Base")
    bodies = set()
    for body_id in range(model.nbody):
        parent = model.body_parentid[body_id]
        while parent != 0:
            if parent == base_id:
                bodies.add(body_id)
                break
            parent = model.body_parentid[parent]
    bodies.discard(base_id)
    return bodies


def evaluate_candidate(
    model,
    data,
    qpos,
    camera_id,
    K,
    zone_mask,
    arm_geoms,
    arm_geom_ids,
    table_geom,
    tag_geoms,
    tag_polygons,
    seg_renderer,
):
    data.qpos[:6] = qpos
    mujoco.mj_forward(model, data)

    contact = None
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == table_geom and c.geom2 in arm_geoms) or (
            c.geom2 == table_geom and c.geom1 in arm_geoms
        ):
            contact = (
                int(c.geom1),
                int(c.geom2),
                float(c.dist),
            )
            break

    seg_renderer.update_scene(data, camera=camera_id)
    segmentation = seg_renderer.render()
    geom_ids = segmentation[..., 0]

    arm_pixels = int(
        np.count_nonzero(zone_mask & np.isin(geom_ids, arm_geom_ids))
    )

    tag_visibility = {}
    for tag_id, geom_id in tag_geoms.items():
        poly, area = tag_polygons[tag_id]
        visible = int(np.count_nonzero((geom_ids == geom_id) & (poly > 0)))
        tag_visibility[tag_id] = visible / area

    return contact, arm_pixels, tag_visibility


def print_pose(qpos):
    print("look pose qpos:", [round(float(v), 6) for v in qpos])


def save_artifacts(model, data, qpos, camera_id):
    data.qpos[:6] = qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    try:
        renderer.update_scene(data, camera=camera_id)
        rgb = renderer.render()
    finally:
        renderer.close()
    cv2.imwrite(TOP_VIEW_PATH, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    with open(LOOK_POSE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"qpos": [float(v) for v in qpos]}, fh, indent=2)
    print(f"saved: {LOOK_POSE_PATH}")
    print(f"saved: {TOP_VIEW_PATH}")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    if model.vis.global_.offwidth < WIDTH:
        model.vis.global_.offwidth = WIDTH
    if model.vis.global_.offheight < HEIGHT:
        model.vis.global_.offheight = HEIGHT

    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera"
    )
    K = build_intrinsics(model, camera_id, WIDTH, HEIGHT)

    arm_bodies = arm_descendants(model)
    arm_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] in arm_bodies
    }
    arm_geom_ids = np.array(sorted(arm_geoms), dtype=np.int32)
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    tag_geoms = {
        tag_id: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"apriltag_{tag_id}"
        )
        for tag_id in range(4)
    }

    data.qpos[:6] = HOME_QPOS
    data.ctrl[:6] = HOME_QPOS
    mujoco.mj_forward(model, data)
    zone_mask = build_zone_mask(data, camera_id, K)
    tag_polygons = build_tag_polygons(data, camera_id, K, tag_geoms)

    seg_renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    seg_renderer.enable_segmentation_rendering()
    try:
        if os.path.exists(MANUAL_PATH):
            print(f"using manual look pose: {MANUAL_PATH}")
            with open(MANUAL_PATH, "r", encoding="utf-8") as fh:
                qpos = np.asarray(json.load(fh)["qpos"], dtype=np.float64)
            contact, arm_pixels, tag_visibility = evaluate_candidate(
                model,
                data,
                qpos,
                camera_id,
                K,
                zone_mask,
                arm_geoms,
                arm_geom_ids,
                table_geom,
                tag_geoms,
                tag_polygons,
                seg_renderer,
            )
            passed = (
                contact is None
                and arm_pixels == 0
                and all(v >= TAG_VISIBLE_MIN for v in tag_visibility.values())
            )
            print_pose(qpos)
            print(f"arm_pixels={arm_pixels}")
            print("contact:", contact)
            print(
                "tag_visibility:",
                {k: round(float(v), 4) for k, v in tag_visibility.items()},
            )
            if not passed:
                print("MANUAL LOOK POSE REJECTED")
                sys.exit(1)
            print("manual look pose passed all checks")
            save_artifacts(model, data, qpos, camera_id)
            return

        ranges = [model.jnt_range[i] for i in range(4)]
        grids = [
            np.linspace(r[0], r[1], n) for r, n in zip(ranges, GRID)
        ]
        candidates = [
            np.array([a, b, c, d, 0.0, HOME_QPOS[5]], dtype=np.float64)
            for a in grids[0]
            for b in grids[1]
            for c in grids[2]
            for d in grids[3]
        ]

        rows = []
        passing = []
        for index, qpos in enumerate(candidates, start=1):
            contact, arm_pixels, tag_visibility = evaluate_candidate(
                model,
                data,
                qpos,
                camera_id,
                K,
                zone_mask,
                arm_geoms,
                arm_geom_ids,
                table_geom,
                tag_geoms,
                tag_polygons,
                seg_renderer,
            )
            ok = (
                contact is None
                and arm_pixels == 0
                and all(v >= TAG_VISIBLE_MIN for v in tag_visibility.values())
            )
            row = (qpos, contact, arm_pixels, tag_visibility, ok)
            rows.append(row)
            if ok:
                passing.append(row)
            if index % 100 == 0:
                print(f"evaluated {index}/{len(candidates)} candidates")

    finally:
        seg_renderer.close()

    if not passing:
        print("NO LOOK POSE PASSED. Best 5 candidates by arm_pixels:")
        for qpos, contact, arm_pixels, tag_visibility, _ in sorted(
            rows, key=lambda r: r[2]
        )[:5]:
            print(
                "  qpos=",
                [round(float(v), 6) for v in qpos],
                "arm_pixels=",
                arm_pixels,
                "contact=",
                contact,
                "tag_visibility=",
                {k: round(float(v), 4) for k, v in tag_visibility.items()},
            )
        sys.exit(1)

    def max_joint_delta(row):
        qpos = row[0]
        return float(np.max(np.abs(qpos[:4] - HOME_QPOS[:4])))

    best = min(passing, key=max_joint_delta)
    qpos = best[0]

    print(f"candidates evaluated: {len(rows)}")
    print(f"candidates passed: {len(passing)}")
    print_pose(qpos)
    print(f"max |qpos[0:4] - HOME_QPOS| = {max_joint_delta(best):.6f} rad")
    print(f"arm_pixels={best[2]}")
    print("contact:", best[1])
    print(
        "tag_visibility:",
        {k: round(float(v), 4) for k, v in best[3].items()},
    )
    save_artifacts(model, data, qpos, camera_id)


if __name__ == "__main__":
    main()
