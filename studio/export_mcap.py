"""Army Perception Studio - MCAP exporter.

Exports one perception snapshot of the MuJoCo sim as a Foxglove .mcap file.
Reuses simulation, calibration, and perception logic from studio/export_scene.py.

Usage (from repo root):
    python studio/export_mcap.py
    python studio/export_mcap.py --layout 2
    python studio/export_mcap.py --noise 0.002
    python studio/export_mcap.py --stride 3 --out studio/army_snapshot.mcap
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Ensure studio/ is importable
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import export_scene as es  # noqa: E402
import foxglove  # noqa: E402
from foxglove.messages import (  # noqa: E402
    Color,
    CubePrimitive,
    KeyValuePair,
    LinePrimitive,
    LinePrimitiveLineType,
    PackedElementField,
    PackedElementFieldNumericType as NT,
    PointCloud,
    Point3,
    Pose,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    SpherePrimitive,
    TextPrimitive,
    TriangleListPrimitive,
    Vector3,
)

OBJ_COLORS = {
    "red_cube": (1.0, 0.2, 0.2),
    "green_cube": (0.2, 0.8, 0.2),
    "yellow_cube": (1.0, 0.9, 0.1),
    "blue_sphere": (0.2, 0.5, 1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layout",
        type=int,
        default=None,
        help="0-4: use one of the random test layouts (default: scene as is)",
    )
    ap.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="depth noise sigma in metres (default 0.0)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=3,
        help="keep 1 pixel out of N in the raw cloud (default 3)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=os.path.join(HERE, "army_snapshot.mcap"),
        help="output MCAP file path (default studio/army_snapshot.mcap)",
    )
    args = ap.parse_args()

    # Load look pose and initialize simulation
    with open(es.LOOK_POSE_PATH, "r", encoding="utf-8") as fh:
        look_qpos = np.asarray(json.load(fh)["qpos"], dtype=np.float64)

    robot = es.RobotAPI(render=False)
    cam = es.SimTopCamera(robot.model, robot.data, depth_noise_sigma=args.noise)
    perception = es.TopCameraPerception()

    try:
        if args.layout is not None:
            addresses = es.object_qpos_addresses(robot.model)
            layout = es.generate_layouts(seed=11)[args.layout]
            es.set_layout(robot.model, robot.data, addresses, layout)

        # 1) Same call the agent uses: moves to the look pose, checks calibration.
        result = es.observe(robot, cam, perception, look_qpos)
        if not result.get("success"):
            sys.exit(f"observe failed: {result.get('error')}")

        # 2) One frame, used for EVERYTHING below so the views agree.
        rgb, depth, K = cam.grab()
        T, residuals_mm = perception.calibrate(rgb, depth, K)
        objects = perception.detect_objects(rgb, depth, K, T)
        X, Y, Z, valid = es.robot_frame_grids(depth, K, T)

        # Sim truth (only exists in simulation).
        truth = {}
        for name in es.OBJECTS:
            bid = es.mujoco.mj_name2id(robot.model, es.mujoco.mjtObj.mjOBJ_BODY, name)
            truth[name] = robot.data.xpos[bid].astype(float).tolist()

        # ---------- 1. /perception/raw_cloud ----------
        s = max(1, args.stride)
        sl = (slice(None, None, s), slice(None, None, s))
        Xs, Ys, Zs, Vs, C = X[sl], Y[sl], Z[sl], valid[sl], rgb[sl]
        keep = (
            Vs
            & (Xs > es.CLOUD_BOX["x"][0])
            & (Xs < es.CLOUD_BOX["x"][1])
            & (Ys > es.CLOUD_BOX["y"][0])
            & (Ys < es.CLOUD_BOX["y"][1])
            & (Zs > es.CLOUD_BOX["z"][0])
            & (Zs < es.CLOUD_BOX["z"][1])
        )
        pos = np.stack([Xs[keep], Ys[keep], Zs[keep]], axis=1).astype(np.float32)
        col = C[keep].astype(np.uint8)

        n_pts = pos.shape[0]
        cloud_arr = np.empty(
            n_pts,
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
                ("a", "u1"),
            ],
        )
        cloud_arr["x"] = pos[:, 0]
        cloud_arr["y"] = pos[:, 1]
        cloud_arr["z"] = pos[:, 2]
        cloud_arr["r"] = col[:, 0]
        cloud_arr["g"] = col[:, 1]
        cloud_arr["b"] = col[:, 2]
        cloud_arr["a"] = 255

        cloud_fields = [
            PackedElementField(name="x", offset=0, type=NT.Float32),
            PackedElementField(name="y", offset=4, type=NT.Float32),
            PackedElementField(name="z", offset=8, type=NT.Float32),
            PackedElementField(name="red", offset=12, type=NT.Uint8),
            PackedElementField(name="green", offset=13, type=NT.Uint8),
            PackedElementField(name="blue", offset=14, type=NT.Uint8),
            PackedElementField(name="alpha", offset=15, type=NT.Uint8),
        ]
        pose_origin = Pose(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        raw_cloud_msg = PointCloud(
            frame_id="robot",
            pose=pose_origin,
            point_stride=16,
            fields=cloud_fields,
            data=cloud_arr.tobytes(),
        )

        # ---------- 2. /perception/tags ----------
        tag_sizes = es.read_tag_sizes()
        tag_entities = []
        for tag_id in sorted(es.TAG_WORLD):
            tag_pos = es.TAG_WORLD[tag_id]
            half_x, half_y = tag_sizes[tag_id]
            res = residuals_mm.get(tag_id, 0.0)

            cube = CubePrimitive(
                pose=Pose(
                    position=Vector3(
                        x=float(tag_pos[0]),
                        y=float(tag_pos[1]),
                        z=float(tag_pos[2]),
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                size=Vector3(x=float(2 * half_x), y=float(2 * half_y), z=0.001),
                color=Color(r=1.0, g=1.0, b=1.0, a=1.0),
            )
            text = TextPrimitive(
                pose=Pose(
                    position=Vector3(
                        x=float(tag_pos[0]),
                        y=float(tag_pos[1]),
                        z=float(tag_pos[2]) + 0.015,
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                billboard=True,
                font_size=0.012,
                scale_invariant=False,
                color=Color(r=1.0, g=1.0, b=1.0, a=1.0),
                text=f"tag {tag_id}  {res:.2f} mm",
            )
            tag_entity = SceneEntity(
                frame_id="robot",
                id=f"tag_{tag_id}",
                frame_locked=False,
                metadata=[],
                arrows=[],
                cubes=[cube],
                spheres=[],
                cylinders=[],
                lines=[],
                triangles=[],
                texts=[text],
                models=[],
            )
            tag_entities.append(tag_entity)
        tags_msg = SceneUpdate(deletions=[], entities=tag_entities)

        # ---------- 3. /perception/objects ----------
        obj_entities = []
        events_objects = []

        for name in es.OBJECTS:
            info = objects[name]
            visible = bool(info["visible"])
            xyz = [float(v) for v in info["xyz"]]
            top_z = float(info["top_z"]) if info.get("top_z") is not None else 0.0

            events_objects.append({
                "name": name,
                "visible": visible,
                "xyz": xyz,
                "top_z": top_z,
            })

            if not visible:
                print(f"WARNING: Object '{name}' visible=False, skipping.")
                continue

            poly = es.footprint(rgb, X, Y, Z, valid, name)
            if not poly or len(poly) < 3:
                print(f"WARNING: Object '{name}' footprint invalid or < 3 points, skipping.")
                continue

            n_poly = len(poly)
            r, g, b = OBJ_COLORS.get(name, (0.8, 0.8, 0.8))

            # Top cap
            top_pts = [Point3(x=float(p[0]), y=float(p[1]), z=top_z) for p in poly]
            top_indices = []
            for k in range(1, n_poly - 1):
                top_indices.extend([0, k, k + 1])
            top_triangles = TriangleListPrimitive(
                pose=pose_origin,
                points=top_pts,
                color=Color(r=r, g=g, b=b, a=1.0),
                colors=[],
                indices=top_indices,
            )

            # Sides
            side_pts = []
            for p in poly:
                side_pts.append(Point3(x=float(p[0]), y=float(p[1]), z=0.0))
                side_pts.append(Point3(x=float(p[0]), y=float(p[1]), z=top_z))
            side_indices = []
            for i in range(n_poly):
                j = (i + 1) % n_poly
                side_indices.extend([2 * i, 2 * j, 2 * j + 1, 2 * i, 2 * j + 1, 2 * i + 1])
            side_triangles = TriangleListPrimitive(
                pose=pose_origin,
                points=side_pts,
                color=Color(r=r, g=g, b=b, a=0.35),
                colors=[],
                indices=side_indices,
            )

            # Text label above object
            obj_text = TextPrimitive(
                pose=Pose(
                    position=Vector3(
                        x=xyz[0],
                        y=xyz[1],
                        z=top_z + 0.02,
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                billboard=True,
                font_size=0.012,
                scale_invariant=False,
                color=Color(r=1.0, g=1.0, b=1.0, a=1.0),
                text=name,
            )

            # Metadata
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            dx_mm = (max(xs) - min(xs)) * 1000.0
            dy_mm = (max(ys) - min(ys)) * 1000.0
            t_pos = truth.get(name, [0.0, 0.0, 0.0])
            err_3d = float(np.linalg.norm(np.array(xyz) - np.array(t_pos)) * 1000.0)

            metadata = [
                KeyValuePair(key="x_mm", value=f"{xyz[0] * 1000.0:.1f}"),
                KeyValuePair(key="y_mm", value=f"{xyz[1] * 1000.0:.1f}"),
                KeyValuePair(key="top_z_mm", value=f"{top_z * 1000.0:.1f}"),
                KeyValuePair(key="center_z_mm", value=f"{xyz[2] * 1000.0:.1f}"),
                KeyValuePair(key="footprint_mm", value=f"{dx_mm:.1f} x {dy_mm:.1f}"),
                KeyValuePair(key="pixels", value=str(info["n_pixels"])),
                KeyValuePair(key="truth_x_mm", value=f"{t_pos[0] * 1000.0:.1f}"),
                KeyValuePair(key="truth_y_mm", value=f"{t_pos[1] * 1000.0:.1f}"),
                KeyValuePair(key="truth_z_mm", value=f"{t_pos[2] * 1000.0:.1f}"),
                KeyValuePair(key="error_3d_mm", value=f"{err_3d:.2f}"),
            ]

            obj_entity = SceneEntity(
                frame_id="robot",
                id=name,
                frame_locked=False,
                metadata=metadata,
                arrows=[],
                cubes=[],
                spheres=[],
                cylinders=[],
                lines=[],
                triangles=[top_triangles, side_triangles],
                texts=[obj_text],
                models=[],
            )
            obj_entities.append(obj_entity)
        objects_msg = SceneUpdate(deletions=[], entities=obj_entities)

        # ---------- 4. /perception/truth ----------
        truth_entities = []
        for name in es.OBJECTS:
            t_pos = truth.get(name, [0.0, 0.0, 0.0])
            truth_sphere = SpherePrimitive(
                pose=Pose(
                    position=Vector3(
                        x=float(t_pos[0]),
                        y=float(t_pos[1]),
                        z=float(t_pos[2]),
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                size=Vector3(x=0.008, y=0.008, z=0.008),
                color=Color(r=1.0, g=1.0, b=1.0, a=0.8),
            )
            truth_entity = SceneEntity(
                frame_id="robot",
                id=name,
                frame_locked=False,
                metadata=[],
                arrows=[],
                cubes=[],
                spheres=[truth_sphere],
                cylinders=[],
                lines=[],
                triangles=[],
                texts=[],
                models=[],
            )
            truth_entities.append(truth_entity)
        truth_msg = SceneUpdate(deletions=[], entities=truth_entities)

        # ---------- 5. /perception/camera_view ----------
        h_img, w_img = depth.shape
        corners = es.view_corners_on_table(K, T, w_img, h_img)
        corner_pts = [Point3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in corners]
        cam_line = LinePrimitive(
            type=LinePrimitiveLineType.LineLoop,
            pose=pose_origin,
            thickness=0.002,
            scale_invariant=False,
            points=corner_pts,
            color=Color(r=0.48, g=0.54, b=0.66, a=1.0),
            colors=[],
            indices=[],
        )
        cam_entity = SceneEntity(
            frame_id="robot",
            id="camera_view_boundary",
            frame_locked=False,
            metadata=[],
            arrows=[],
            cubes=[],
            spheres=[],
            cylinders=[],
            lines=[cam_line],
            triangles=[],
            texts=[],
            models=[],
        )
        camera_view_msg = SceneUpdate(deletions=[], entities=[cam_entity])

        # ---------- 6. /army/events ----------
        events_msg = {
            "event": "perception_snapshot",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layout": args.layout,
            "noise_m": args.noise,
            "calibration_residual_mm_max": float(max(residuals_mm.values())),
            "objects": events_objects,
        }

    finally:
        cam.close()
        robot.close()

    # Write MCAP file: log the whole snapshot twice (t0 and t0 + 1s)
    t0 = time.time_ns()
    t1 = t0 + 1_000_000_000  # 1 second later

    with foxglove.open_mcap(args.out, allow_overwrite=True):
        for ts in (t0, t1):
            foxglove.log("/perception/raw_cloud", raw_cloud_msg, log_time=ts)
            foxglove.log("/perception/tags", tags_msg, log_time=ts)
            foxglove.log("/perception/objects", objects_msg, log_time=ts)
            foxglove.log("/perception/truth", truth_msg, log_time=ts)
            foxglove.log("/perception/camera_view", camera_view_msg, log_time=ts)
            foxglove.log("/army/events", events_msg, log_time=ts)

    out_size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({out_size_mb:.1f} MB)")
    print(f"cloud points: {n_pts}   max tag residual: {max(residuals_mm.values()):.2f} mm")
    for name in es.OBJECTS:
        info = objects[name]
        state = "ok " if info["visible"] else "MISSING"
        xyz_str = (
            f"xyz(mm)=({info['xyz'][0]*1000.0:.1f}, {info['xyz'][1]*1000.0:.1f}, {info['xyz'][2]*1000.0:.1f})"
            if info["visible"]
            else ""
        )
        print(f"  {name:<12} {state} {xyz_str}")


if __name__ == "__main__":
    main()
