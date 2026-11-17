"""Vision core for the overhead top camera.

This module performs camera calibration from AprilTags and colour-based object
detection.  It is deliberately free of any MuJoCo dependency: callers supply
RGB, metric z-depth and intrinsics.

Coordinate convention: camera points are OpenCV-style (x right, y down, z
forward).  The calibration returns a rigid transform mapping camera points into
the robot/world frame.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np
from pupil_apriltags import Detector

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "third_party",
    "so_arm100",
    "interactive_scene.xml",
)

HALF_HEIGHT = {
    "red_cube": 0.012,
    "green_cube": 0.012,
    "yellow_cube": 0.012,
    "blue_sphere": 0.014,
}

# One HSV range per colour.  OpenCV hue is 0..179.  A range whose lower hue is
# greater than its upper hue wraps through red (e.g. 172..8).
HSV_RANGES = {
    "red_cube": ((172, 90, 60), (8, 255, 255)),
    "green_cube": ((55, 90, 60), (82, 255, 255)),
    "yellow_cube": ((18, 90, 60), (42, 255, 255)),
    "blue_sphere": ((88, 90, 60), (110, 255, 255)),
}

TAG_IDS = (0, 1, 2, 3)
MAX_TAG_RESIDUAL_MM = 3.0


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_array(self):
        return np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)


def _read_tag_world():
    """Read tag top-surface centres from the XML once and freeze them."""
    fallback = {
        0: (-0.28, -0.07, 0.0011),
        1: (0.28, -0.07, 0.0011),
        2: (-0.28, -0.33, 0.0011),
        3: (0.28, -0.33, 0.0011),
    }
    try:
        root = ET.parse(_MODEL_PATH).getroot()
    except (OSError, ET.ParseError):
        return dict(fallback)

    tags = {}
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if not name.startswith("apriltag_"):
            continue
        try:
            tag_id = int(name.rsplit("_", 1)[1])
            pos = [float(v) for v in geom.get("pos").split()]
            size = [float(v) for v in geom.get("size").split()]
        except (ValueError, AttributeError, IndexError):
            continue
        tags[tag_id] = (pos[0], pos[1], pos[2] + size[2])
    return tags if len(tags) == 4 else dict(fallback)


TAG_WORLD = _read_tag_world()


def deproject(u, v, z, K):
    """Pixel + metric z-depth -> point in the OpenCV camera frame."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return np.stack(
        [(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z], axis=-1
    )


def kabsch(cam_points, robot_points):
    """Rigid transform (R, t) mapping camera points to robot points, no scale."""
    P = np.asarray(cam_points, dtype=np.float64)
    Q = np.asarray(robot_points, dtype=np.float64)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cQ - R @ cP
    return R, t


def _patch_median(depth, u, v, radius=2):
    h, w = depth.shape
    u0 = int(round(float(u)))
    v0 = int(round(float(v)))
    u_lo, u_hi = max(0, u0 - radius), min(w, u0 + radius + 1)
    v_lo, v_hi = max(0, v0 - radius), min(h, v0 + radius + 1)
    patch = np.asarray(depth[v_lo:v_hi, u_lo:u_hi], dtype=np.float64)
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def _color_mask(hsv, name):
    lo, hi = HSV_RANGES[name]
    s_min = min(lo[1], hi[1])
    v_min = min(lo[2], hi[2])
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0
    low = cv2.inRange(
        hsv,
        np.array([lo[0], s_min, v_min]),
        np.array([179, 255, 255]),
    )
    high = cv2.inRange(
        hsv,
        np.array([0, s_min, v_min]),
        np.array([hi[0], 255, 255]),
    )
    return cv2.bitwise_or(low, high) > 0


class TopCameraPerception:
    def __init__(self, detector=None):
        self._detector = detector or Detector(families="tag36h11")

    def calibrate(self, rgb, depth, K):
        """Fit camera->robot from AprilTags.

        Returns (T_cam_to_robot 4x4, residuals_mm dict tag_id -> mm).
        Raises ValueError for fewer than four tags or any residual > 3 mm.
        """
        gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        detections = self._detector.detect(gray)

        cam_points = []
        robot_points = []
        tag_ids = []
        for detection in detections:
            tag_id = int(detection.tag_id)
            if tag_id not in TAG_WORLD:
                continue
            u, v = detection.center
            z = _patch_median(depth, u, v, radius=2)
            if z is None:
                continue
            cam_points.append(deproject(u, v, z, K))
            robot_points.append(np.asarray(TAG_WORLD[tag_id], dtype=np.float64))
            tag_ids.append(tag_id)

        if len(tag_ids) < 4:
            raise ValueError(
                f"AprilTag calibration needs 4 tags, only {len(tag_ids)} usable "
                f"(ids {sorted(tag_ids)})."
            )

        R, t = kabsch(cam_points, robot_points)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        residuals_mm = {}
        for tag_id, pc, pr in zip(tag_ids, cam_points, robot_points):
            residuals_mm[tag_id] = float(np.linalg.norm(R @ pc + t - pr) * 1000.0)

        worst_id = max(residuals_mm, key=residuals_mm.get)
        if residuals_mm[worst_id] > MAX_TAG_RESIDUAL_MM:
            raise ValueError(
                f"AprilTag calibration residual {residuals_mm[worst_id]:.2f} mm "
                f"(tag {worst_id}) exceeds {MAX_TAG_RESIDUAL_MM:.1f} mm."
            )

        return T, residuals_mm

    def detect_objects(self, rgb, depth, K, T):
        """Detect the four coloured objects in the robot frame."""
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth = np.asarray(depth, dtype=np.float64)
        depth = cv2.GaussianBlur(depth, (7, 7), 0)
        T = np.asarray(T, dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        valid = np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)
        z_cam = depth[ys, xs]
        cam_pts = deproject(xs, ys, z_cam, K)
        robot_pts = cam_pts @ R.T + t

        shape = depth.shape
        x_robot = np.full(shape, np.nan, dtype=np.float64)
        y_robot = np.full(shape, np.nan, dtype=np.float64)
        z_robot = np.full(shape, np.nan, dtype=np.float64)
        x_robot[ys, xs] = robot_pts[:, 0]
        y_robot[ys, xs] = robot_pts[:, 1]
        z_robot[ys, xs] = robot_pts[:, 2]

        kernel = np.ones((3, 3), dtype=np.uint8)
        result = {}
        for name in HALF_HEIGHT:
            mask = _color_mask(hsv, name).astype(np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=2) > 0
            above = eroded & np.isfinite(z_robot) & (z_robot > 0.008)
            n_pixels = int(np.count_nonzero(above))

            if n_pixels < 30:
                result[name] = {
                    "xyz": None,
                    "top_z": None,
                    "n_pixels": n_pixels,
                    "visible": False,
                }
                continue

            z_vals = z_robot[above]
            ref = float(np.percentile(z_vals, 80))
            keep = z_vals >= ref - 0.006

            top_ref = float(np.percentile(z_vals, 99))
            refined = keep & (z_vals >= top_ref - 0.002)
            if int(np.count_nonzero(refined)) >= 20:
                keep = refined

            x_vals = x_robot[above][keep]
            y_vals = y_robot[above][keep]
            if name == "blue_sphere":
                top_z = float(np.percentile(z_vals, 95))
            else:
                top_z = float(np.median(z_vals[keep]))

            xyz = [
                float(np.mean(x_vals)),
                float(np.mean(y_vals)),
                float(top_z - HALF_HEIGHT[name]),
            ]
            result[name] = {
                "xyz": xyz,
                "top_z": top_z,
                "n_pixels": n_pixels,
                "visible": True,
            }

        return result
