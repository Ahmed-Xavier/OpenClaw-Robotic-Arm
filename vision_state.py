"""Camera-only observation pipeline for the robot."""

import numpy as np

LOOK_POSE_TOL_RAD = 0.03


def observe(robot, cam, perception, look_qpos):
    """Move to the look pose (unless holding) and return camera-detected objects."""
    look_qpos = np.asarray(look_qpos, dtype=np.float64).reshape(-1)
    if look_qpos.size < 5:
        return {
            "success": False,
            "error": {
                "code": "INVALID_LOOK_POSE",
                "message": "look_qpos must contain at least 5 joint values.",
            },
        }

    if getattr(robot, "_holding", False):
        source = "held_pose"
    else:
        target_ctrl = robot.data.ctrl.copy()
        target_ctrl[:5] = look_qpos[:5]
        robot._step_to_ctrl(target_ctrl, steps=80)

        delta = np.abs(np.asarray(robot.data.qpos[:5], dtype=np.float64) - look_qpos[:5])
        for _ in range(6):
            if float(np.max(delta)) < LOOK_POSE_TOL_RAD:
                break
            robot._step_to_ctrl(target_ctrl, steps=10)
            delta = np.abs(
                np.asarray(robot.data.qpos[:5], dtype=np.float64) - look_qpos[:5]
            )

        if float(np.max(delta)) >= LOOK_POSE_TOL_RAD:
            return {
                "success": False,
                "error": {
                    "code": "LOOK_POSE_NOT_REACHED",
                    "message": (
                        f"Look pose not reached: max joint error "
                        f"{float(np.max(delta)):.4f} rad >= {LOOK_POSE_TOL_RAD:.2f} rad "
                        f"(per-joint {np.round(delta, 4).tolist()})."
                    ),
                },
            }
        source = "look_pose"

    previous_busy = getattr(robot, "_busy_moving", False)
    robot._busy_moving = True
    try:
        rgb, depth, K = cam.grab()
    finally:
        robot._busy_moving = previous_busy

    try:
        T, residuals_mm = perception.calibrate(rgb, depth, K)
    except Exception as exc:
        return {
            "success": False,
            "error": {"code": "CALIBRATION_FAILED", "message": str(exc)},
        }

    objects = perception.detect_objects(rgb, depth, K, T)
    residual_max = float(max(residuals_mm.values())) if residuals_mm else float("inf")

    return {
        "success": True,
        "objects": objects,
        "calibration_residual_mm_max": residual_max,
        "source": source,
    }
