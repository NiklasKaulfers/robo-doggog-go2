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
3. **Walks after the target.** `FollowPerson` keeps a fixed distance, steers toward the
   person (proportional control), detours around obstacles using a remembered LiDAR
   map, slows down on stairs, and when the person leaves the view it heads where they
   were going, looks around, and finally roams.
4. **Stays safe.** Movement only goes through the Go2 obstacle avoidance service. Any
   remote controller input pauses the program, and Ctrl+C hands control back cleanly.

## Architecture

```
                         ┌──────────────────────────── main.cpp ────────────────────────────┐
                         │ args, signal handlers, DDS init, stand up, run the trick         │
                         └───────────────────────────────┬──────────────────────────────────┘
                                                         │
                                          tricks/FollowPerson  (control loop)
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
| `src/core/` | Robot access. **`Control`** wraps the Unitree SDK: robot state (pose, IMU) and remote controller input arrive over DDS; movement commands go out via `ObstacleAvoidClient` (never directly via `SportClient::Move`, which would bypass avoidance). It tracks remote activity, hands control back when the remote is used or the program stops, and detects the explore button combo (L1 + R1 + Y). **`ObstacleMap`** copies the small LiDAR height map into a larger grid that remembers terrain for a few minutes, and offers free-distance queries, A*-style route planning and reachable/frontier spots. |
| `src/ml/` | Perception, no robot dependencies except the MoveNet wrapper. **`PersonDetection`** is the only type shared between detector and recognition code. **`FaceRecognizer`** (YuNet + SFace, OpenCV DNN) holds a gallery of known faces and returns match / mismatch / unknown. **`Appearance`** describes clothes as histograms and keeps a set of views (anchor views learned while the face matched are never forgotten). **`movenet/`** runs the pose models through LiteRT. |
| `src/tracking/` | **`TargetTracker`** is the decision maker: it associates detections to tracks, accumulates face and clothing evidence, picks and keeps the target, drops it when clothes or face contradict, takes it over when a face matches someone else, and predicts where a lost person went (in the odometry frame). **`CameraGeometry.h`** converts pixels to angle and distance. |
| `src/tricks/` | Behaviours built on `Control`. **`FollowPerson`** is the real application. **`Explore`** roams to new spots. **`TurnToPerson`** and **`SitStandApproach`** are earlier experiments (gesture and turn demos), kept but not used by `main`. |
| `tools/` | Offline tools that need only OpenCV: **`facecheck`** (quality of reference photos) and **`trackertest`** (self-test of the tracking logic with drawn people). |
| `res/` | Models: `face_detection_yunet_2022mar.onnx`, `face_recognition_sface_2021dec.onnx`, `movenet-multipose.tflite`, `movenet-singlepose.tflite`. `res/target/` (git-ignored) holds the reference photos. |

### Dependency split

The CMake project builds two parts:

- **`recognition`** (static library): `FaceRecognizer`, `Appearance`, `TargetTracker`,
  `CameraGeometry`, `PersonDetection`. Needs only OpenCV, so it is shared by the app
  and the offline tools and can be built on a laptop with `-DFOLLOW_TOOLS_ONLY=ON`.
- **`app`**: everything else. Needs the Unitree SDK 2 (`../sdk/install`) and LiteRT
  (`../sdk/litert`), which exist only in the Jetson devcontainer.

### Control loop (`FollowPerson::run`)

Every iteration:

1. Grab a camera frame and run the pose detector; convert to `PersonDetection`s.
2. Feed them, the frame and the robot pose to `TargetTracker::update`, which returns a
   `Status`: `Searching`, `Tracking` or `Lost`, plus the target's predicted angle and
   distance.
3. If the remote is in use, send nothing. If the explore combo was pressed, toggle
   exploring.
4. Otherwise choose movement:
   - **Tracking**: forward speed from the distance error (target distance 0.5 m), yaw
     from the angle error; `planHeading` replaces the heading with a detour when the
     direct path is blocked, using `ObstacleMap`.
   - **Lost**: walk toward the predicted position, look toward the side the person left,
     after 10 s roam with `Explore` while the detector keeps looking.
   - **Searching**: stand still and wait.
5. `drive()` sends the command; on detected stairs (body pitch/height change with
   hysteresis) it limits it to slow, straight, forward movement.

### Target selection in short

- With reference photos, nobody is followed until a face matches. Without photos, the
  first stable person is followed and their face and clothes are learned live.
- While following, continuity of the box keeps the track; clothes are checked every
  frame, faces whenever visible.
- A clear face mismatch drops the target at once; half a second of clothes that don't
  fit also drops it. A face match on another person takes the target over.
- Thresholds are constants at the top of `TargetTracker.cpp`, `FaceRecognizer.h`,
  `FollowPerson.cpp` and `Appearance.cpp`.

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
```

Binaries: `app` (robot), `facecheck`, `trackertest`.
