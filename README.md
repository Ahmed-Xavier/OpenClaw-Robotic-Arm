# OpenClaw Embodied Arm (Sim)

Give OpenClaw a body: a UR5e in Gazebo, controlled through an OpenClaw skill,
following the CLI-bridge pattern from a real working hardware project.
UPDATE: ROSCLAW plugin MIGHT BE HELPFUL https://github.com/ros-claw/rosclaw
i found this repo that doesn't use ubuntu, it's interesting https://github.com/Seeed-Projects/awesome-openclaw-hardware-projects/blob/main/07-robotics/control-soarm101.md
This isn't a from-scratch design — it's two existing, working repos stitched
together with a translation layer in between:

| Piece | Source | Role |
|---|---|---|
| Simulated body | [`UniversalRobots/Universal_Robots_ROS2_GZ_Simulation`](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation) | UR5e + Gazebo + `ros2_control` + MoveIt 2, ready to run |
| Agent↔robot bridge pattern | [`proknowdiy/AI-Robotic-Arm-RDK-S100-OpenClaw`](https://github.com/proknowdiy/AI-Robotic-Arm-RDK-S100-OpenClaw) | Proven OpenClaw-skill → CLI → robot-driver architecture, re-pointed at MoveIt 2 instead of an ESP32 |

---

## 1. What's actually in the UR5e sim repo

Checked directly, not from the repo's own marketing copy.

- Package: `ur_simulation_gz`, launch files `ur_sim_control.launch.py` (raw
  `ros2_control`, no planning) and `ur_sim_moveit.launch.py` (MoveIt 2 +
  Gazebo, this is the one we want).
- **Branch matters.** The repo's `HEAD`/default branch (`ros2`) tracks
  **Lyrical/Rolling**, not what you're running. There's a dedicated **`jazzy`
  branch** that matches ROS 2 Jazzy on Ubuntu 24.04 — same distro you already
  have working on Echo. Clone `-b jazzy`, not the default branch, or you'll
  fight dependency mismatches for no reason.
- Standard colcon workspace flow: clone into `src/ur_simulation_gz`, `rosdep
  install`, `colcon build --symlink-install`, source, then
  `ros2 launch ur_simulation_gz ur_sim_moveit.launch.py`.
- No perception, no task logic, no agent hooks — it's purely the simulated
  body + planner. Everything above that layer is on us.

## 2. What's actually in the RDK-S100 OpenClaw repo

This is a small, real, working project (4DOF arm, ESP32-C3, RDK S100) — not
just an idea. The architecture is worth stealing exactly because it's already
proven end-to-end on hardware:

```
python/
  arm_api.py         # CLI entrypoint — dispatches string commands to task_executor
  task_executor.py    # business logic: precondition checks, then calls robot_controller
  robot_controller.py  # hardware driver: JSON-over-serial to the ESP32
  scene_detector.py    # OpenCV: fixed pixel regions -> slot contents (cube colors)
openclaw_skill/
  robotic-arm.md        # OpenClaw skill definition (YAML frontmatter + instructions)
```

The actual bridge mechanism: **OpenClaw doesn't call Python functions
directly.** The skill file tells the agent to shell out —
`python3 arm_api.py pick A` — and read stdout. `arm_api.py` is a thin
dispatcher (`scene`, `status`, `pick <slot>`, `drop <side>`, `move <slot>
<side>`, `sort_red`, `sort_green`, `home`) over `task_executor.py`, which
does the actual precondition checking (already holding a cube? slot empty?)
before touching `robot_controller.py`, which talks to the ESP32 over a JSON
serial protocol (`{"action": "pick_cube", "source": "A"}` → wait for
`"Done"` or `"ERROR:..."`).

The one rule worth copying verbatim into our skill file is this one, straight
from `robotic-arm.md`:

> Before any action that depends on slot or bin contents, re-run the scene
> command. Act on the live scene output, not memory of the previous result.

That's the right defense against an agent hallucinating stale world state —
we want the same rule for object poses in Gazebo.

**What we are *not* reusing literally:** the A/B/C fixed-pixel-region slot
detection and the 4DOF serial protocol are specific to that hardware tray.
Our "driver layer" is MoveIt 2 action calls instead of JSON-over-serial, and
our "scene layer" is RGB-D + object detection instead of three hardcoded
pixel boxes.

---

## 3. Architecture (adapted)

```
                    USER
                     │
                     ▼
             OpenClaw skill (arm.md)
                     │  shells out, same pattern as robotic-arm.md
                     ▼
              robot_api.py (CLI)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  task_executor.py          scene_detector.py
  (precondition checks,     (RGB-D → object
   re-check-scene rule)      poses, replaces
        │                    fixed pixel regions)
        ▼
  moveit_driver.py
  (replaces robot_controller.py:
   MoveIt 2 action client instead
   of JSON-over-serial)
        │
        ▼
   ROS 2 / MoveIt 2 / ros2_control
        │
        ▼
   Gazebo — UR5e simulation
```

Same three-layer separation as the RDK project (CLI dispatcher → task logic
→ hardware driver), same "OpenClaw shells out and reads stdout" bridge —
just the bottom two layers swapped for sim/MoveIt 2 instead of
serial/ESP32.

---

## 4. Repo structure

```
openclaw-embodied-arm/
├── README.md
├── ur_ws/                       # colcon workspace
│   └── src/
│       └── ur_simulation_gz/    # cloned from jazzy branch, upstream, untouched
├── robot_api/
│   ├── arm_api.py                # CLI dispatcher (same shape as proknowdiy's)
│   ├── task_executor.py          # precondition checks + re-check-scene rule
│   ├── moveit_driver.py          # MoveIt 2 action client (replaces robot_controller.py)
│   └── scene_detector.py         # RGB-D object detection (replaces fixed pixel regions)
├── openclaw_skill/
│   └── robotic-arm.md            # adapted skill definition
└── docs/
    └── setup.md
```

---

## 5. Roadmap (scoped down — no games, no chess)

Games and multi-object recognition are explicitly parked, not on the
critical path. Order of operations:

- [ ] **Phase 0 — Environment**: clone `ur_simulation_gz` on the `jazzy`
      branch, build workspace, confirm `ur_sim_moveit.launch.py` brings up
      UR5e + MoveIt 2 in Gazebo with no errors.
- [ ] **Phase 1 — Manual sim control**: drive the arm through MoveIt 2's own
      interface (RViz motion planning plugin or a test script) before any
      agent is involved. Confirm IK and collision checking work.
- [ ] **Phase 2 — Scene objects**: add a table + a couple of colored cubes to
      the Gazebo world.
- [ ] **Phase 3 — Perception**: RGB-D camera in sim, publish 3D object poses
      (start with color/contour segmentation like the RDK project — no need
      for a heavy detector against known-color cubes).
- [ ] **Phase 4 — `robot_api` layer**: build `arm_api.py` /
      `task_executor.py` / `moveit_driver.py`, CLI-testable without OpenClaw
      in the loop first (mirrors how the RDK repo lets you run
      `python3 main.py` standalone before wiring OpenClaw at all).
- [ ] **Phase 5 — OpenClaw skill**: adapt `robotic-arm.md`, point it at
      `arm_api.py`, keep the re-check-scene rule.
- [ ] **Phase 6 — First embodied task**: "pick up the red cube" end-to-end,
      observe → reason → act → verify.

Multimodal local model note: since Phase 3 needs the LLM to reason over
scene state (not just call tools blindly on text), the model backing
OpenClaw needs vision input, not just text. Worth deciding early whether
that model runs locally (same pattern as Echo's qwen3.5:4b on the RTX 4050
over Tailscale) or against a hosted multimodal API, since it changes the
latency budget of the observe→act loop.

---

## 6. Setup notes

```bash
export COLCON_WS=~/workspaces/openclaw_arm
mkdir -p $COLCON_WS/src
cd $COLCON_WS
git clone -b jazzy https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git src/ur_simulation_gz
rosdep update && rosdep install --ignore-src --from-paths src -y
colcon build --symlink-install
source install/setup.bash
ros2 launch ur_simulation_gz ur_sim_moveit.launch.py
```

If that comes up clean, Phase 0 is done and Phase 1 (manual MoveIt 2 control)
starts.
