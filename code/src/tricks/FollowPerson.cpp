#include "FollowPerson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>
#include "opencv2/imgcodecs.hpp"
#include "../ml/movenet/Multipose.h"

using namespace robo::sdk::examples;
using namespace robo::sdk::ml;
using namespace robo::sdk::tracking;

// stairs are detected by the body tilting forward/backward. Entering and leaving use
// different thresholds (hysteresis) so the robot doesn't toggle on every step edge.
static constexpr float STAIR_PITCH_ENTER = 0.14f;   // ~8 degrees
static constexpr float STAIR_PITCH_EXIT = 0.07f;    // ~4 degrees
static constexpr int STAIR_ENTER_SAMPLES = 3;       // 0.3s tilted -> on stairs
static constexpr int STAIR_EXIT_SAMPLES = 10;       // 1.0s level -> off stairs
// the body also rises/drops by roughly one step height per step
static constexpr float STAIR_HEIGHT_CHANGE = 0.12f;
// on stairs: slow, mostly straight and never backwards
static constexpr float STAIR_MAX_SPEED_UP = 0.35f;
static constexpr float STAIR_MAX_SPEED_DOWN = 0.25f;
static constexpr float STAIR_MAX_YAW = 0.2f;
// tilted sideways this far means the robot is about to slip off a step
static constexpr float STAIR_MAX_ROLL = 0.35f;      // ~20 degrees

// pose detections below this confidence are ignored
static constexpr float MIN_CONFIDENCE = 0.15f;
// how often a debug frame is written
static constexpr auto DEBUG_FRAME_INTERVAL = std::chrono::milliseconds(1000);
// watchdog (s): odometry older than this stops the robot. (A stale height map is handled
// by the FollowPlanner: it only turns then.)
static constexpr float STATE_TIMEOUT = 0.5f;
// fixed period of the run loop; the sample counts above (e.g. 3 samples = 0.3 s) assume it
static constexpr auto LOOP_PERIOD = std::chrono::milliseconds(100);

FollowPerson::FollowPerson(core::Control& control, std::string targetImages, std::string debugDir)
    : roboControl(control), targetImages(std::move(targetImages)), planner(obstacleMap, std::random_device {}()),
      debugDir(std::move(debugDir)) {
    videoClient.Init();
    mapSubscriber.init();
}

void FollowPerson::run() {
    runWithMultipose();
}

void FollowPerson::collectDetections(const movenet::Multipose& detector, std::vector<PersonDetection>& out) {
    out.clear();
    const cv::Mat& frame = detector.frame();
    const cv::Rect2f bounds(0, 0, frame.cols, frame.rows);

    for (int d = 0; d < detector.modelDetectionCount(); d++) {
        const auto box = detector.getBox(d);
        if (box.confidence < MIN_CONFIDENCE) continue;

        PersonDetection person;
        person.score = box.confidence;
        const cv::Point2f topLeft = detector.toFrame(box.xmin, box.ymin);
        const cv::Point2f bottomRight = detector.toFrame(box.xmax, box.ymax);
        person.box = cv::Rect2f(topLeft, bottomRight) & bounds;
        if (person.box.width <= 0 || person.box.height <= 0) continue;

        for (int k = 0; k < PersonDetection::KEYPOINT_COUNT; k++) {
            // Movenet's keypoints are stored as (y, x, confidence): Keypoint::x is the
            // row and Keypoint::y the column
            const auto keypoint = detector.getKeypoint(d, k);
            person.keypoints[k] = detector.toFrame(keypoint.y, keypoint.x);
            person.confidence[k] = keypoint.confidence;
        }
        out.push_back(person);
    }
}

void FollowPerson::runWithMultipose() {
    movenet::Multipose detector;
    if (!detector.init()) return;

    if (!planner.init()) return;
    if (!targetImages.empty()) {
        if (!planner.loadReferencePhotos(targetImages)) {
            std::cerr << "[FollowPerson] no usable face found in the photos in " << targetImages << std::endl;
            return;
        }
        std::cout << "[FollowPerson] waiting to see the face of the person from the photos" << std::endl;
    } else {
        std::cout << "[FollowPerson] no photos given, following the first person seen" << std::endl;
    }

    if (!debugDir.empty()) {
        std::error_code error;
        std::filesystem::create_directories(debugDir, error);
        if (error) {
            std::cerr << "[FollowPerson] cannot create " << debugDir << ": " << error.message() << ", debug frames off" << std::endl;
            debugDir.clear();
        } else {
            std::cout << "[FollowPerson] writing debug frames to " << debugDir << std::endl;
        }
    }

    std::vector<PersonDetection> detections;
    int stairSamples = 0;
    // set from the first robot state message: before it positionZ() is 0, which would
    // look like a height change and trigger stair mode
    float levelHeight = 0;
    bool levelHeightValid = false;
    auto nextTick = std::chrono::steady_clock::now();
    // sensor failures are logged once when they start and once when they end
    bool stateStale = false;
    bool cameraFailed = false;

    while (!core::Control::stopRequested) {
        // fixed period: sleep only the rest of it. A tick that overran resyncs instead of
        // running a burst of catch-up ticks
        nextTick += LOOP_PERIOD;
        const auto tickStart = std::chrono::steady_clock::now();
        if (nextTick > tickStart) std::this_thread::sleep_until(nextTick);
        else nextTick = tickStart;

        if (!levelHeightValid && std::isfinite(roboControl.stateAge())) {
            levelHeight = roboControl.positionZ();
            levelHeightValid = true;
        }
        if (levelHeightValid) updateStairMode(stairSamples, levelHeight);

        if (roboControl.exploreComboPressed()) {
            exploring = !exploring;
            if (exploring) {
                explorer.reset();
                std::cout << "[FollowPerson] exploring" << std::endl;
            } else {
                std::cout << "[FollowPerson] stopped exploring, following again" << std::endl;
            }
            // operator reset: afterwards the next person (or the one in the photos) is followed
            planner.reset();
        }

        // the remote controller always wins: don't send anything while it is used. Time
        // under remote control doesn't count toward the search
        if (roboControl.controllerActive()) {
            roboControl.releaseControl();
            moving = false;
            planner.pause();
            continue;
        }

        // odometry watchdog: without fresh pose and IMU data nothing below can be trusted
        const bool stateOld = roboControl.stateAge() > STATE_TIMEOUT;
        if (stateOld != stateStale) {
            stateStale = stateOld;
            if (stateStale) std::cerr << "[FollowPerson] no fresh robot state, stopping" << std::endl;
            else std::cout << "[FollowPerson] robot state is updating again" << std::endl;
        }
        if (stateStale) {
            stop();
            planner.pause();
            continue;
        }

        // roams until the explore combo is pressed again
        if (exploring) {
            float x, z;
            explorer.step(x, z);
            drive(x, z);
            planner.pause();
            continue;
        }

        // camera or detector failure: stop instead of walking on with the last command
        const int32_t imageStatus = videoClient.GetImageSample(imgData);
        const bool imageOk = imageStatus == 0 && !imgData.empty();
        if (!imageOk || !detector.run(imgData)) {
            if (!cameraFailed) {
                if (!imageOk) std::cerr << "[FollowPerson] no camera image (status " << imageStatus << "), stopping" << std::endl;
                else std::cerr << "[FollowPerson] error in detector.run(), stopping" << std::endl;
            }
            cameraFailed = true;
            stop();
            planner.pause();
            continue;
        }
        if (cameraFailed) {
            std::cout << "[FollowPerson] camera and detector working again" << std::endl;
            cameraFailed = false;
        }

        const auto now = std::chrono::steady_clock::now();
        const RobotPose pose {roboControl.positionX(), roboControl.positionY(), roboControl.yaw()};
        const cv::Mat& frame = detector.frame();

        collectDetections(detector, detections);
        const auto& step = planner.step(detections, frame, pose, now);
        if (!debugDir.empty()) saveDebugFrame(frame, now);
        // every phase goes through drive(): stair limits and obstacle avoidance apply
        drive(step.x, step.z);
    }
    // leaving the loop: don't leave the last command running
    stop();
}

void FollowPerson::saveDebugFrame(const cv::Mat& frame, std::chrono::steady_clock::time_point now) {
    if (frame.empty() || now - lastDebugFrame < DEBUG_FRAME_INTERVAL) return;
    lastDebugFrame = now;

    cv::Mat annotated = frame.clone();
    planner.drawDebug(annotated);
    std::ostringstream name;
    name << "frame_" << std::setw(5) << std::setfill('0') << debugFrameCount++ << ".jpg";
    const auto path = std::filesystem::path(debugDir) / name.str();
    if (!cv::imwrite(path.string(), annotated, {cv::IMWRITE_JPEG_QUALITY, 80}))
        std::cerr << "[FollowPerson] could not write " << path << std::endl;
}

void FollowPerson::drive(float x, float z) {
    if (stairMode) {
        if (std::abs(roboControl.roll()) > STAIR_MAX_ROLL) {
            if (moving) std::cerr << "[FollowPerson] tilted sideways on stairs, stopping" << std::endl;
            stop();
            return;
        }

        // a nose-down body means the robot is going down the stairs
        const float maxSpeed = roboControl.pitch() > 0 ? STAIR_MAX_SPEED_DOWN : STAIR_MAX_SPEED_UP;
        x = std::clamp(x, 0.0f, maxSpeed);
        z = std::clamp(z, -STAIR_MAX_YAW, +STAIR_MAX_YAW);
    }

    if (x != 0 || z != 0) {
        roboControl.moveRelative(x, z);
        moving = true;
    } else {
        stop();
    }
}

void FollowPerson::updateStairMode(int& stairSamples, float& levelHeight) {
    const float pitch = std::abs(roboControl.pitch());
    const float height = roboControl.positionZ();

    if (!stairMode) {
        const bool tilted = pitch > STAIR_PITCH_ENTER;
        const bool heightChanged = std::abs(height - levelHeight) > STAIR_HEIGHT_CHANGE;
        stairSamples = (tilted || heightChanged) ? stairSamples + 1 : 0;

        if (stairSamples >= STAIR_ENTER_SAMPLES) {
            std::cout << "[FollowPerson] stairs detected, moving carefully" << std::endl;
            stairMode = true;
            stairSamples = 0;
        } else if (stairSamples == 0) {
            levelHeight = height;
        }
    } else {
        stairSamples = pitch < STAIR_PITCH_EXIT ? stairSamples + 1 : 0;

        if (stairSamples >= STAIR_EXIT_SAMPLES) {
            std::cout << "[FollowPerson] back on level ground" << std::endl;
            stairMode = false;
            stairSamples = 0;
            levelHeight = height;
        }
    }
}

void FollowPerson::stop() {
    // only send the stop once, repeating it would block the remote controller
    if (!moving) return;
    // keep `moving` set when the stop failed so the next tick retries
    if (roboControl.stopMovement()) moving = false;
}
