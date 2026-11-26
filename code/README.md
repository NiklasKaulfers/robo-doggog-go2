# Robo Doggo Logic (`code/`)

C++17 program that makes a Unitree Go2 quadruped follow one specific person. It runs
on the Go2's onboard Jetson, reads the robot's front camera, finds and recognizes the
person, and walks after them. Everything the robot does goes through its built-in
obstacle avoidance, and the remote controller always takes priority.

For usage, photo tips, tuning and debug frames see the [top-level README](../README.md).
This file describes what the code does and how it is put together.

## What it does

1. **Sees people.** Each camera frame goes through MoveNet MultiPose (TFLite, run with
   LiteRT), which returns a box and 17 body keypoints per person.
2. **Decides who is the target.** A `TargetTracker` tracks everyone in view from frame
   to frame and collects evidence about who is who:
   - **Face**: YuNet finds faces inside each person's head region, SFace turns them into
     embeddings that are compared with reference photos (or with faces learned live).
   - **Clothes**: color histograms of torso and legs, taken at the keypoints. They work
     from behind, where the face is not visible.
3. **Walks after the target.** `FollowPlanner` keeps a fixed distance, steers toward the
   person (proportional control) and detours around obstacles using a remembered LiDAR
   map. When the person leaves the view it searches in phases: it heads where they were
   going (Pursue), turns once on the spot (LookAround), roams a growing area around
   where they were last seen (Sweep), and after 10 minutes stops and watches (GaveUp).
   `FollowPerson` connects it to the robot and slows down on stairs.
4. **Keeps a log.** Every log line is also saved with timestamps, level and component in
   a local SQLite file on the dog, ready to be uploaded when there is a connection.
5. **Stays safe.** Movement only goes through the Go2 obstacle avoidance service. Any
   remote controller input pauses the program, and Ctrl+C hands control back cleanly.

## Architecture

```
                         ┌──────────────────────────── main.cpp ────────────────────────────┐
                         │ args, signal handlers, DDS init, stand up, run the trick         │
                         └───────────────────────────────┬──────────────────────────────────┘
                                                         │
                                          tricks/FollowPerson  (control loop, robot glue)
                                                         │
                                          follow/FollowPlanner (phases, chasing, searching)
        ┌───────────────────────────┬────────────────────┼─────────────────────┬──────────────────────┐
        ▼                           ▼                    ▼                     ▼                      ▼
 Go2 VideoClient          ml/movenet/Multipose   tracking/TargetTracker   core/ObstacleMap       tricks/Explore
 (JPEG camera frames)     (pose detector)        (who to follow)          (LiDAR memory,         (roaming)
                                 │                  │         │           route planning)
                                 ▼                  ▼         ▼
                          ml/PersonDetection   ml/FaceRecognizer  ml/Appearance
                          (boxes+keypoints)    (YuNet + SFace)    (clothing histograms)

                         core/Control: SportModeState + WirelessController in,
                         obstacle avoidance commands out
```

### Layers

| Folder | Role |
| --- | --- |
| `src/core/` | Robot access. **`Control`** wraps the Unitree SDK: robot state (pose, IMU) and remote controller input arrive over DDS; movement commands go out via `ObstacleAvoidClient` (never directly via `SportClient::Move`, which would bypass avoidance). It tracks remote activity, hands control back when the remote is used or the program stops, and detects the explore button combo (L1 + R1 + Y). **`ObstacleMap`** copies the small LiDAR height map into a larger grid that remembers terrain for a few minutes, and offers free-distance queries, A*-style route planning (with a mild preference for the previous route, and able to end next to the person at the goal) and reachable/frontier spots. It is plain C++ without SDK types: frames come in as `HeightMapFrame` through `integrate()`, `secondsSinceUpdate()` tells how old the map is and `checkFrame()` warns when the map and odometry disagree. **`ObstacleMapSubscriber`** is the small SDK part that subscribes to `rt/utlidar/height_map_array` and feeds it in. |
| `src/ml/` | Perception, no robot dependencies except the MoveNet wrapper. **`PersonDetection`** is the only type shared between detector and recognition code. **`FaceRecognizer`** (YuNet + SFace, OpenCV DNN) holds a gallery of known faces and returns match / mismatch / unknown. **`Appearance`** describes clothes as histograms and keeps a set of views (anchor views learned while the face matched are never forgotten). **`movenet/`** runs the pose models through LiteRT. |
| `src/tracking/` | **`TargetTracker`** is the decision maker: it associates detections to tracks, accumulates face and clothing evidence, picks and keeps the target, drops it when clothes or face contradict, takes it over when a face matches someone else, and predicts where a lost person went (in the odometry frame). **`CameraGeometry.h`** converts pixels to angle and distance. |
| `src/follow/` | **`FollowPlanner`** decides following and searching, without a robot: one `step()` per camera frame takes the detections, the frame, the robot pose and the time, and returns forward/yaw speed, the phase (`WaitingForTarget`, `Following`, `Pursue`, `LookAround`, `Sweep`, `GaveUp`) and the tracker's status. It owns the `TargetTracker` and an `ExplorePlanner` for the Sweep (with a `SearchFocus`: goals only inside a circle, spots the way the person walked score higher). `pause()` keeps time the robot wasn't driven by it out of the search timers, `reset()` is the operator reset. Search constants are at the top of `FollowPlanner.cpp`. |
| `src/tricks/` | Behaviours built on `Control`. **`FollowPerson`** is the real application: camera, pose detector, remote priority, explore combo, watchdogs, stairs, debug frames; the decisions are `FollowPlanner`'s. **`Explore`** roams to new spots. **`TurnToPerson`** and **`SitStandApproach`** are earlier experiments (gesture and turn demos), kept but not used by `main`. |
| `src/logging/` | Structured log, see [ADR 0001](../docs/adr/0001-local-sqlite-log-outbox.md). Every log line is a **`LogRecord`** (wall and steady timestamps, session id, level, component, message, key/value fields). The **`Logger`** prints it as `[Component] message` like before (debug/info on stdout, warn/error on stderr) and queues it; a background thread saves the queue to the **`LogStore`**, a local SQLite outbox (`--log-db`, default `logs/robo.db`) that survives crashes and restarts and keeps a "pushed" flag per record. The store is capped (`--log-max-mb`): uploaded records are deleted first, then the oldest. Code without a logger passed in calls `logging::info/warn/...`, which use the logger `main` installs, or print to the console when there is none (tools, tests). **`Uploader`** + `shipPending` are the seam for a future upload to DynamoDB/MongoDB; there is no network code yet. If the store fails, logging continues on the console only. |
| `tools/` | Offline tools that need no robot: **`facecheck`** (quality of reference photos), **`logtest`** (self-test of the logging library on a real SQLite file: round trip, sessions, pushed flag, durability, concurrent producers, queue overflow, the size cap, a broken store, `shipPending`, console format), **`trackertest`** (self-test of the tracking logic with drawn people), **`followtest`** (self-test of `FollowPlanner` with a simulated robot, simulated people drawn into camera frames and synthetic terrain: following, bystanders, obstacles, Pursue, LookAround, Sweep, GaveUp, finding the target again in every phase, paused time, operator reset), **`appearancetest`** (self-test of the clothes model), **`exploretest`** (self-test of `ExplorePlanner` with a simulated robot: stale map, frame mismatch, unreachable goals, timeouts, obstacle slow-down, a long run in a walled room) and **`maptest`** (self-test of `ObstacleMap` with synthetic height maps: walls, gaps, stairs, drops, a person at the goal, memory, hysteresis; `--bench` also times the planner). |
| `res/` | Models: `face_detection_yunet_2022mar.onnx`, `face_recognition_sface_2021dec.onnx`, `movenet-multipose.tflite`, `movenet-singlepose.tflite`. `res/target/` (git-ignored) holds the reference photos. |

### Dependency split

The CMake project builds five parts:

- **`logging`** (static library): `Logger`, `LogStore`, `Uploader`. Needs only SQLite: the
  system's if CMake finds it (macOS has it), otherwise the official amalgamation is
  downloaded at configure time (pinned by hash) and built in, so nothing is installed on
  the Jetson. Every other part links it. Built with `-DFOLLOW_TOOLS_ONLY=ON`, tested by `logtest`.
- **`recognition`** (static library): `FaceRecognizer`, `Appearance`, `TargetTracker`,
  `CameraGeometry`, `PersonDetection`. Needs only OpenCV, so it is shared by the app
  and the offline tools and can be built on a laptop with `-DFOLLOW_TOOLS_ONLY=ON`.
- **`navigation`** (static library): `ObstacleMap`, `ExplorePlanner`. Needs only the standard library,
  also built with `-DFOLLOW_TOOLS_ONLY=ON` and tested by `maptest` and `exploretest`.
- **`following`** (static library): `FollowPlanner`, on top of `recognition` and `navigation`.
  Also built with `-DFOLLOW_TOOLS_ONLY=ON` and tested by `followtest`.
- **`app`**: everything else, including `ObstacleMapSubscriber` (the DDS side of the map). Needs the Unitree SDK 2 (`../sdk/install`) and LiteRT
  (`../sdk/litert`), which exist only in the Jetson devcontainer.

### Control loop (`FollowPerson::run`)

Every iteration:

1. If the explore combo was pressed, toggle exploring and reset the planner (operator
   reset). If the remote is in use, send nothing. If the robot state (odometry/IMU) is
   older than 0.5 s, or the camera or detector fails, stop. While exploring, `Explore`
   drives. In all these cases `FollowPlanner::pause()` keeps the time out of the search.
2. Otherwise grab a camera frame and run the pose detector; convert to
   `PersonDetection`s and pass them, the frame, the robot pose and the time to
   `FollowPlanner::step`. It feeds the `TargetTracker` and chooses movement by phase:
   - **WaitingForTarget**: no target chosen yet, stand still.
   - **Following**: the target is in view (or was a few frames ago). Forward speed from
     the distance error (target distance 0.5 m), yaw from the angle error; the heading is
     replaced with a detour when the direct path is blocked, using `ObstacleMap`. Forward
     speed is limited by the free distance both along that heading and straight ahead, and
     scaled by its cosine so the robot turns onto a detour before walking. With a stale
     height map (> 1 s) it only turns. Caps: 0.6 m/s, 0.6 rad/s.
   - **Pursue** (tracker reports Lost): chase the predicted position the same way, until
     within 0.6 m of it or for 15 s.
   - **LookAround**: turn on the spot at 0.4 rad/s toward the side the person left,
     until the odometry yaw says one full turn is done.
   - **Sweep**: `ExplorePlanner` roams with a search focus around where the person was
     last seen (3 m, growing 0.15 m/s up to 20 m, biased toward their walking direction).
     Its caps apply (0.4 m/s, 0.6 rad/s); on a stale map it stops.
   - **GaveUp**: after 600 s of searching (paused time excluded), stand still.
   In every search phase, the target being recognized again means Following at once.
3. `drive()` sends the command; on detected stairs (body pitch/height change with
   hysteresis) it limits it to slow, straight, forward movement, in every phase.

### Target selection in short

- With reference photos, nobody is followed until a face matches. Without photos, the
  first stable person is followed and their face and clothes are learned live. That
  "first person" rule applies only once per run: after a target was chosen, only a
  face, clothes or the short "showed up where they went" window bring one back.
  `TargetTracker::dropTarget()` (the operator reset) re-arms it and, without photos,
  forgets the learned face and clothes so the next person can become the target.
- While following, continuity of the box keeps the track; clothes are checked every
  frame, faces whenever visible.
- A clear face mismatch drops the target at once; half a second of clothes that don't
  fit also drops it. A face match on another person takes the target over.
- Thresholds are constants at the top of `TargetTracker.cpp`, `FaceRecognizer.h`,
  `FollowPlanner.cpp`, `FollowPerson.cpp` and `Appearance.cpp`.

## Building

On the Jetson (devcontainer with OpenCV, Unitree SDK, LiteRT):

```
cmake -S . -B build -G Ninja
cmake --build build
./build/app
```

On a laptop with only OpenCV (offline tools):

```
cmake -S . -B build-tools -DFOLLOW_TOOLS_ONLY=ON
cmake --build build-tools
./build-tools/trackertest     # run from this folder, needs ./res
./build-tools/maptest         # needs nothing, exit code 0 when all checks pass
./build-tools/exploretest     # needs nothing, ~40 s (simulated room runs)
./build-tools/appearancetest  # needs nothing
./build-tools/followtest      # run from this folder, needs ./res, a few seconds
./build-tools/logtest         # needs nothing, under a second
```

Binaries: `app` (robot), `logtest`, `facecheck`, `trackertest`, `appearancetest`, `maptest`, `exploretest`, `followtest`.
