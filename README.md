# Robo Doggo Logic
for the dawg

The Go2 follows one person. It recognizes them by their face (from photos you give it)
and, once it has seen them, by their clothes, so it keeps following them from behind
and tells them apart from other people. Movement always goes through the robot's
obstacle avoidance, and the remote controller always has priority.

## Running

Build on the Go2's Jetson (inside the devcontainer), then from the `code` folder:

```
./build/app                       # follows the person in ./res/target, or the first person seen
./build/app /path/to/photos       # photos of the person to follow
./build/app --no-target           # ignore ./res/target, follow the first person seen
./build/app --debug ./debug       # also write an annotated camera frame per second
```

### Photos of the person

Put a few jpg/png photos of the person in `code/res/target/` (the folder is ignored by
git). What works best:

- 3 to 8 photos, face at least 40 px wide, ideally a few hundred
- different angles (frontal, slightly left/right) and lighting (indoor, daylight)
- only that person up front; if there are other faces, the largest face is used
- no sunglasses, no heavy motion blur

Check them without the robot (also builds on a laptop with OpenCV, see below):

```
./build/facecheck ./res/target                    # how well the photos agree with each other
./build/facecheck ./res/target ./debug            # which faces in the debug frames would match
./build/facecheck ./res/target photo.jpg --out ./checked
```

All photos should be similar to each other with at least 0.363 (OpenCV's SFace
threshold). A photo that is below that with every other one is probably a bad photo or
another person, and the app warns about it when loading.

### What happens

- **With photos**: the robot waits until it sees a face that matches. Only then does it
  start following. While following, it learns how that person's clothes look on its
  camera (views learned while the face matched are kept for the whole run). From then
  on the clothes recognize them from behind. Whenever their face is visible it is
  checked again.
- **Without photos**: the first person that stays in view for a moment is followed, and
  their face (when they look at the robot) and clothes are learned from the camera.
  From then on they are the one being recognized, just like with photos.
- **Someone steps in front / people cross**: the clothes of the person being followed
  are compared every frame. If they stop looking like the target's for half a second,
  the target is dropped and searched for. A face that clearly isn't the target's drops
  them at once, and a matching face on someone else takes the target over.
- **Out of view**: the robot heads where the person was walking (predicted from their
  last position and speed), looks around there, and after 10 s roams around. They are
  recognized again by their face or clothes anywhere in view; for the first 2 s also
  by simply showing up in the direction they went, if their clothes don't contradict it.
- **Remote**: any stick or button pauses the program until the remote has been idle for
  1.5 s. L1 + R1 + Y toggles exploring (roaming on its own); pressing it again goes
  back to following whoever is recognized next.

### Debug frames

`--debug DIR` writes `frame_00000.jpg`, ... once a second with everyone in view:
track id, face evidence (-1..1) and the latest face similarity, clothing similarity,
the head region searched for a face (yellow), the face found (green = match, red =
mismatch), and the torso/legs regions the clothes are taken from. The target has a
thick green box. The log prints a status line every 5 s with the same numbers, use
them for tuning.

### Tuning

The thresholds are constants at the top of these files:

| What | Where |
| --- | --- |
| face match / mismatch thresholds, minimum face sizes | `code/src/ml/FaceRecognizer.h` |
| clothing: how alike to keep following (0.5), to find again (0.7), to learn (0.6) | `code/src/tracking/TargetTracker.cpp` |
| face evidence needed to be the person / to be rejected | `code/src/tracking/TargetTracker.cpp` |
| camera field of view and distance calibration | `code/src/tracking/CameraGeometry.h` |
| following distance, speeds, obstacle detours, stairs | `code/src/tricks/FollowPerson.cpp` |
| histogram layout and body regions of the clothes | `code/src/ml/Appearance.cpp` |

## Code layout

```
code/src/core/        Control (movement via obstacle avoidance, remote priority), ObstacleMap (LiDAR memory, route planning)
code/src/ml/          PersonDetection (a person in frame pixels), FaceRecognizer (YuNet + SFace),
                      Appearance (clothing descriptor and model), movenet/ (pose detector via LiteRT)
code/src/tracking/    TargetTracker (who to follow: tracks everyone, accumulates face/clothing
                      evidence, picks and keeps the target), CameraGeometry
code/src/tricks/      FollowPerson (the walking), Explore (roaming), TurnToPerson, SitStandApproach
code/tools/           facecheck (check photos), trackertest, appearancetest, maptest, exploretest (offline self-tests)
code/res/             models: movenet-multipose.tflite, face_detection_yunet_2022mar.onnx, face_recognition_sface_2021dec.onnx
```

## Building

On the Jetson (devcontainer, OpenCV, Unitree SDK and LiteRT present):

```
cmake -S code -B code/build -G Ninja
cmake --build code/build
```

The recognition code only needs OpenCV, so the offline tools also build on a laptop:

```
cmake -S code -B build-tools -DFOLLOW_TOOLS_ONLY=ON
cmake --build build-tools
cd code && ../build-tools/trackertest      # self-test, needs ./res for the face models
```
