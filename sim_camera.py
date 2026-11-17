"""MuJoCo adapter that produces (rgb, depth, intrinsics) for the top camera."""

import numpy as np
import mujoco

from perception import Intrinsics


class SimTopCamera:
    def __init__(
        self,
        model,
        data,
        camera="top_camera",
        width=1280,
        height=960,
        depth_noise_sigma=0.0,
    ):
        self.model = model
        self.data = data
        self.camera = camera
        self.width = int(width)
        self.height = int(height)
        self.depth_noise_sigma = float(depth_noise_sigma)

        if model.vis.global_.offwidth < self.width:
            model.vis.global_.offwidth = self.width
        if model.vis.global_.offheight < self.height:
            model.vis.global_.offheight = self.height

        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        if self.camera_id == -1:
            raise ValueError(f"Camera '{camera}' not found in model.")

        fovy = float(model.cam_fovy[self.camera_id])
        fy = (self.height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        self.K = Intrinsics(
            fx=fy, fy=fy, cx=self.width / 2.0, cy=self.height / 2.0
        )

        self._rgb_renderer = mujoco.Renderer(
            model, height=self.height, width=self.width
        )
        self._depth_renderer = mujoco.Renderer(
            model, height=self.height, width=self.width
        )
        self._depth_renderer.enable_depth_rendering()
        self._rng = np.random.default_rng(0)

    def grab(self):
        """Return (rgb uint8 HxWx3, z-depth float64 HxW in metres, K)."""
        self._rgb_renderer.update_scene(self.data, camera=self.camera)
        rgb = np.asarray(self._rgb_renderer.render(), dtype=np.uint8).copy()

        self._depth_renderer.update_scene(self.data, camera=self.camera)
        depth = np.asarray(self._depth_renderer.render(), dtype=np.float64).copy()

        if self.depth_noise_sigma > 0.0:
            valid = np.isfinite(depth) & (depth > 0)
            noise = self._rng.normal(
                0.0, self.depth_noise_sigma, size=int(np.count_nonzero(valid))
            )
            depth[valid] += noise

        return rgb, depth, self.K

    def close(self):
        self._rgb_renderer.close()
        self._depth_renderer.close()
