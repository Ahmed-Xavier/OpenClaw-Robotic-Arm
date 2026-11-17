"""Read-only Foxglove telemetry bridge for the Army MuJoCo simulation.

Run through server.py with ARMY_FOXGLOVE=1. The bridge reuses the existing
SimTopCamera and TopCameraPerception pipeline; it does not create another
MuJoCo model or simulation.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
import foxglove
from foxglove import Channel
from foxglove.channels import (
    CameraCalibrationChannel,
    PointCloudChannel,
    RawImageChannel,
    SceneUpdateChannel,
)
from foxglove.messages import (
    CameraCalibration,
    Color,
    CubePrimitive,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    RawImage,
    SceneEntity,
    SceneEntityDeletion,
    SceneUpdate,
    SpherePrimitive,
    Timestamp,
    Vector3,
)

from perception import TopCameraPerception, deproject
from sim_camera import SimTopCamera

log = logging.getLogger("army.foxglove")


class FoxgloveBridge:
    """Publish camera, state, perception and 3D markers to Foxglove."""

    OBJECT_SIZES = {
        "red_cube": (0.024, 0.024, 0.024),
        "green_cube": (0.024, 0.024, 0.024),
        "yellow_cube": (0.024, 0.024, 0.024),
        "blue_sphere": (0.028, 0.028, 0.028),
    }
    OBJECT_COLORS = {
        "red_cube": Color(r=0.95, g=0.12, b=0.12, a=0.85),
        "green_cube": Color(r=0.12, g=0.85, b=0.25, a=0.85),
        "yellow_cube": Color(r=0.98, g=0.78, b=0.10, a=0.85),
        "blue_sphere": Color(r=0.12, g=0.42, b=0.98, a=0.85),
    }

    def __init__(
        self,
        robot,
        host: str | None = None,
        port: int | None = None,
        publish_hz: float | None = None,
        image_width: int = 640,
        image_height: int = 480,
    ):
        self.robot = robot
        self.host = host or os.getenv("FOXGLOVE_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("FOXGLOVE_PORT", "8766"))
        self.publish_hz = float(
            publish_hz or os.getenv("ARMY_FOXGLOVE_HZ", "8")
        )
        self.camera = SimTopCamera(
            robot.model,
            robot.data,
            camera="top_camera",
            width=image_width,
            height=image_height,
        )
        self.perception = TopCameraPerception()

        self.rgb_channel = RawImageChannel("/army/camera/top/rgb")
        self.depth_channel = RawImageChannel("/army/camera/top/depth")
        self.calibration_channel = CameraCalibrationChannel(
            "/army/camera/top/calibration"
        )
        self.state_channel = Channel("/army/state", message_encoding="json")
        self.detections_channel = Channel(
            "/army/perception/objects", message_encoding="json"
        )
        self.scene_channel = SceneUpdateChannel("/army/perception/scene")
        self.pointcloud_channel = PointCloudChannel("/army/perception/points")

        self._server = None
        self._thread = None
        self._stop = threading.Event()
        self._last_error_log = 0.0

    def start(self):
        """Start the local Foxglove WebSocket server and publisher thread."""
        if self._thread and self._thread.is_alive():
            return
        self._server = foxglove.start_server(host=self.host, port=self.port)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._publish_loop, name="army-foxglove", daemon=True
        )
        self._thread.start()
        log.info("Foxglove WebSocket listening at ws://%s:%d", self.host, self.port)

    def _publish_loop(self):
        period = 1.0 / max(self.publish_hz, 0.5)
        calibration_sent = False

        while not self._stop.is_set():
            cycle_start = time.perf_counter()
            now_ns = time.time_ns()

            try:
                # MuJoCo MjData is shared with the physics thread. Hold its lock
                # while rendering and taking the state snapshot.
                with self.robot._data_lock:
                    rgb, depth, K = self.camera.grab()
                    state = self.robot.get_state()
                    sim_time = float(self.robot.data.time)

                depth32 = np.nan_to_num(
                    np.asarray(depth, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                height, width = rgb.shape[:2]

                self.rgb_channel.log(
                    RawImage(
                        frame_id="top_camera",
                        width=width,
                        height=height,
                        encoding="rgb8",
                        step=width * 3,
                        data=np.ascontiguousarray(rgb).tobytes(),
                    ),
                    log_time=now_ns,
                )
                self.depth_channel.log(
                    RawImage(
                        frame_id="top_camera",
                        width=width,
                        height=height,
                        encoding="32FC1",
                        step=width * 4,
                        data=np.ascontiguousarray(depth32).tobytes(),
                    ),
                    log_time=now_ns,
                )

                if not calibration_sent:
                    self.calibration_channel.log(
                        CameraCalibration(
                            frame_id="top_camera",
                            width=width,
                            height=height,
                            K=[
                                float(K.fx), 0.0, float(K.cx),
                                0.0, float(K.fy), float(K.cy),
                                0.0, 0.0, 1.0,
                            ],
                            P=[
                                float(K.fx), 0.0, float(K.cx), 0.0,
                                0.0, float(K.fy), float(K.cy), 0.0,
                                0.0, 0.0, 1.0, 0.0,
                            ],
                        ),
                        log_time=now_ns,
                    )
                    calibration_sent = True

                try:
                    transform, residuals = self.perception.calibrate(
                        rgb, depth, K
                    )
                    objects = self.perception.detect_objects(
                        rgb, depth, K, transform
                    )
                    perception_result = {
                        "ok": True,
                        "frame_id": "world",
                        "sim_time": sim_time,
                        "calibration_residual_mm": residuals,
                        "objects": objects,
                    }
                    self.detections_channel.log(
                        perception_result, log_time=now_ns
                    )
                    self.scene_channel.log(
                        self._make_scene(objects), log_time=now_ns
                    )
                    self.pointcloud_channel.log(
                        self._make_pointcloud(rgb, depth, K, transform),
                        log_time=now_ns,
                    )
                except Exception as exc:
                    self.detections_channel.log(
                        {
                            "ok": False,
                            "sim_time": sim_time,
                            "error": str(exc),
                            "objects": {},
                        },
                        log_time=now_ns,
                    )
                    # Empty updates let previous short-lived markers expire.
                    self.scene_channel.log(self._make_scene({}), log_time=now_ns)

                state_payload = dict(state)
                state_payload["sim_time"] = sim_time
                state_payload["publish_time_ns"] = now_ns
                self.state_channel.log(state_payload, log_time=now_ns)

            except Exception:
                now = time.monotonic()
                if now - self._last_error_log > 5.0:
                    log.exception("Foxglove publish cycle failed")
                    self._last_error_log = now

            remaining = period - (time.perf_counter() - cycle_start)
            self._stop.wait(max(0.0, remaining))


    @staticmethod
    def _make_pointcloud(rgb, depth, K, transform) -> PointCloud:
        """Create a compact colored XYZ point cloud in the calibrated world frame."""
        # Downsample the 640x480 stream to keep WebSocket bandwidth reasonable.
        stride = 4
        rgb_s = np.asarray(rgb, dtype=np.uint8)[::stride, ::stride]
        depth_s = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
        h, w = depth_s.shape
        vv, uu = np.mgrid[0:h, 0:w]
        uu = uu.reshape(-1) * stride
        vv = vv.reshape(-1) * stride
        zz = depth_s.reshape(-1)
        valid = np.isfinite(zz) & (zz > 0.0)
        camera_points = deproject(uu[valid], vv[valid], zz[valid], K)
        T = np.asarray(transform, dtype=np.float64)
        world_points = camera_points @ T[:3, :3].T + T[:3, 3]
        colors = rgb_s.reshape(-1, 3)[valid]

        packed = np.empty(
            world_points.shape[0],
            dtype=np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
            ], align=False),
        )
        packed["x"] = world_points[:, 0]
        packed["y"] = world_points[:, 1]
        packed["z"] = world_points[:, 2]
        packed["r"] = colors[:, 0]
        packed["g"] = colors[:, 1]
        packed["b"] = colors[:, 2]
        packed["a"] = 255

        numeric = PackedElementFieldNumericType
        fields = [
            PackedElementField(name="x", offset=0, type=numeric.Float32),
            PackedElementField(name="y", offset=4, type=numeric.Float32),
            PackedElementField(name="z", offset=8, type=numeric.Float32),
            PackedElementField(name="r", offset=12, type=numeric.Uint8),
            PackedElementField(name="g", offset=13, type=numeric.Uint8),
            PackedElementField(name="b", offset=14, type=numeric.Uint8),
            PackedElementField(name="a", offset=15, type=numeric.Uint8),
        ]
        return PointCloud(
            frame_id="world",
            point_stride=16,
            fields=fields,
            data=packed.tobytes(),
        )

    def _make_scene(self, objects: dict) -> SceneUpdate:
        entities = []
        for name, detection in objects.items():
            xyz = detection.get("xyz")
            if not detection.get("visible") or xyz is None:
                continue
            size = self.OBJECT_SIZES.get(name, (0.025, 0.025, 0.025))
            pose = Pose(
                position=Vector3(
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2] + size[2] / 2.0),
                )
            )
            color = self.OBJECT_COLORS.get(
                name, Color(r=0.9, g=0.9, b=0.9, a=0.8)
            )
            if name == "blue_sphere":
                entity = SceneEntity(
                    id=f"detected_{name}",
                    frame_id="world",
                    spheres=[
                        SpherePrimitive(
                            pose=pose,
                            size=Vector3(x=size[0], y=size[1], z=size[2]),
                            color=color,
                        )
                    ],
                )
            else:
                entity = SceneEntity(
                    id=f"detected_{name}",
                    frame_id="world",
                    cubes=[
                        CubePrimitive(
                            pose=pose,
                            size=Vector3(x=size[0], y=size[1], z=size[2]),
                            color=color,
                        )
                    ],
                )
            entities.append(entity)
        # Delete known IDs first so detections that disappear do not leave stale markers.
        deletions = [
            SceneEntityDeletion(timestamp=Timestamp.now(), id=f"detected_{name}")
            for name in self.OBJECT_SIZES
        ]
        return SceneUpdate(deletions=deletions, entities=entities)

    def stop(self):
        """Stop publishing and release camera/server resources."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.camera.close()
        if self._server is not None:
            self._server.stop()
            self._server = None
        log.info("Foxglove bridge stopped")
