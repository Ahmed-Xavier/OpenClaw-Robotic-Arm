# Army headless simulation and Foxglove

## Overview

Army can run the existing MuJoCo SO-ARM100 scene without opening the interactive
MuJoCo viewer. The same `RobotAPI` model and `MjData` are used by the REST API,
the existing top-camera RGB/depth pipeline, AprilTag calibration, object
perception, and the Foxglove publisher.

No ROS 2 installation is required.

## Install

Activate the Python environment used by Army, then run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Foxglove Python SDK provides Windows x64 wheels for supported Python
versions (Python 3.10+). MuJoCo remains pinned to the version already used by
the project.

## Run headless

From the repository root:

```powershell
$env:ARMY_RENDER = "0"
$env:ARMY_FOXGLOVE = "1"
$env:FOXGLOVE_HOST = "127.0.0.1"
$env:FOXGLOVE_PORT = "8766"
$env:ARMY_FOXGLOVE_HZ = "8"
python server.py
```

Two local services start:

- Army REST API: `http://127.0.0.1:8765`
- Foxglove WebSocket: `ws://127.0.0.1:8766`

In Foxglove, choose **Open connection → Foxglove WebSocket** and enter
`ws://127.0.0.1:8766`.

## Topics

| Topic | Schema | Purpose |
|---|---|---|
| `/army/camera/top/rgb` | `foxglove.RawImage` | Top-camera RGB image |
| `/army/camera/top/depth` | `foxglove.RawImage` | Metric float32 depth image |
| `/army/camera/top/calibration` | `foxglove.CameraCalibration` | Camera intrinsics |
| `/army/state` | JSON | Robot joints, gripper, objects, holding state and simulation time |
| `/army/perception/objects` | JSON | AprilTag calibration result and detected object XYZ |
| `/army/perception/scene` | `foxglove.SceneUpdate` | 3D object markers in the calibrated world frame |

The publisher currently runs at 8 Hz by default. Set `ARMY_FOXGLOVE_HZ`
to change the rate. Camera rendering and perception use the existing
`SimTopCamera` and `TopCameraPerception` modules.

## Optional MuJoCo GUI

To open the interactive viewer as well:

```powershell
$env:ARMY_RENDER = "1"
python server.py
```

To disable Foxglove while keeping the REST API:

```powershell
$env:ARMY_FOXGLOVE = "0"
python server.py
```

## Notes and current limits

- The Foxglove bridge is read-only. It does not issue robot commands.
- Headless mode still advances MuJoCo physics in the background.
- The 3D scene currently shows detected objects, not a full articulated robot
  model. MuJoCo XML is not automatically a Foxglove robot description; robot
  link visualization needs an explicit model/transform export step.
- Depth is streamed as a float32 image. Select the depth topic in an Image
  panel; the 3D point-cloud view can be added once a calibrated point-cloud
  publisher and camera/world transform are enabled.
- The bridge uses the existing AprilTag detector. If all four tags cannot be
  detected, object markers are withheld and the error is published on the
  detections topic.
- The simulation, camera renderer, and perception load share the same
  `MjData`. If rendering is too heavy, reduce the image size in
  `FoxgloveBridge` or lower `ARMY_FOXGLOVE_HZ`.
