#include "FollowPerson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>
#include "opencv2/imgcodecs.hpp"
#include "../ml/movenet/Multipose.h"
#include "../tracking/CameraGeometry.h"

using namespace robo::sdk::examples;
using namespace robo::sdk::ml;
using namespace robo::sdk::tracking;

static constexpr float TARGET_DISTANCE = 0.5f;

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

static constexpr float TURN_GAIN = 1.0f;
// route planning: how far ahead the path to the person must be free
static constexpr float PLAN_LOOKAHEAD = 1.5f;
// the person's legs show up as an obstacle in the height map, stop checking before them
static constexpr float PERSON_MARGIN = 0.4f;
// closer than this to an obstacle the robot only turns, it doesn't walk forward
static constexpr float MIN_CLEARANCE = 0.35f;
// detours are tried in steps of ~10 degrees up to ~70 degrees to either side
static constexpr float DETOUR_STEP = 0.17f;
static constexpr int DETOUR_STEPS = 7;
// a planned route is followed by steering toward the point this far along it
static constexpr float PATH_LOOKAHEAD = 0.8f;

// pose detections below this confidence are ignored
static constexpr float MIN_CONFIDENCE = 0.15f;
// once at the predicted spot, look around toward the side the person left
static constexpr float SEARCH_ARRIVED = 0.6f;
static constexpr float SEARCH_TURN_SPEED = 0.4f;
// after this the robot roams around semi-randomly until the person shows up again
static constexpr float SEARCH_TIMEOUT = 10.0f;
// how often a debug frame is written
static constexpr auto DEBUG_FRAME_INTERVAL = std::chrono::milliseconds(1000);

FollowPerson::FollowPerson(core::Control& control, std::string targetImages, std::string debugDir)
    : roboControl(control), targetImages(std::move(targetImages)), debugDir(std::move(debugDir)) {
    videoClient.Init();
    obstacleMap.init();
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

    if (!tracker.init()) return;
    if (!targetImages.empty()) {
        if (!tracker.loadReferencePhotos(targetImages)) {
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
    float levelHeight = roboControl.positionZ();

    while (!core::Control::stopRequested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        updateStairMode(stairSamples, levelHeight);

        if (roboControl.exploreComboPressed()) {
            exploring = !exploring;
            if (exploring) {
                explorer.reset();
                std::cout << "[FollowPerson] exploring" << std::endl;
            } else {
                std::cout << "[FollowPerson] stopped exploring, following again" << std::endl;
            }
            // afterwards follow whoever is recognized next
            tracker.dropTarget();
            roamSearching = false;
        }

        // the remote controller always wins: don't send anything while it is used
        if (roboControl.controllerActive()) {
            roboControl.releaseControl();
            moving = false;
            continue;
        }

        // roams until the explore combo is pressed again
        if (exploring) {
            float x, z;
            explorer.step(x, z);
            drive(x, z);
            continue;
        }

        videoClient.GetImageSample(imgData);
        if (!detector.run(imgData)) {
            std::cerr << "[FollowPerson] error in detector.run()" << std::endl;
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        const RobotPose pose {roboControl.positionX(), roboControl.positionY(), roboControl.yaw()};
        const cv::Mat& frame = detector.frame();

        collectDetections(detector, detections);
        const auto& target = tracker.update(detections, frame, pose, now);
        if (!debugDir.empty()) saveDebugFrame(frame, now);

        float distance, personAngle;
        if (target.visible) {
            roamSearching = false;
            distance = CameraGeometry::distance(target.target.box, frame.cols);
            personAngle = CameraGeometry::angle(target.target.box, frame.cols);
        } else if (target.state == TargetTracker::State::Searching) {
            // nobody to follow (yet): wait where we are
            stop();
            continue;
        } else {
            // not where they were heading: walk around semi-randomly, the detector keeps
            // running every step, so the person is followed again as soon as they show up
            if (target.secondsSinceSeen > SEARCH_TIMEOUT) {
                if (!roamSearching) {
                    std::cout << "[FollowPerson] person not found nearby, roaming around to find them" << std::endl;
                    explorer.reset();
                    roamSearching = true;
                }
                float x, z;
                explorer.step(x, z);
                drive(x, z);
                continue;
            }

            // head for the predicted position; it's outside the camera view, so turning
            // toward it means looking in the direction the person left
            distance = target.predictedDistance;
            personAngle = target.predictedAngle;
        }

        // proportional control so the motion is smooth instead of walking straight and
        // suddenly turning once the person leaves a band around the image center
        // (x: move toward the person if too far, away if too close)
        float x = std::clamp(0.5f * (distance - TARGET_DISTANCE), -0.6f, +0.6f);

        // when walking toward the person, take a detour around obstacles in the way
        float heading = personAngle;
        if (x > 0) {
            float clearance = 0;
            heading = planHeading(personAngle, distance, clearance);
            // slow down in front of obstacles and only turn if nothing is free
            x = std::min(x, std::max(0.0f, 0.5f * (clearance - MIN_CLEARANCE)));
        }

        // z: rotate toward the chosen heading, proportional to how far off it is.
        // with this there is no "perfect center" needed for the rotation to start
        float z = std::clamp(TURN_GAIN * heading, -0.6f, +0.6f);

        // arrived where the person should be but still can't see them: look around,
        // starting with the side they disappeared to
        if (!target.visible && target.predictedDistance < SEARCH_ARRIVED) {
            x = 0;
            z = target.lostSide * SEARCH_TURN_SPEED;
        }

        drive(x, z);
    }
}

void FollowPerson::saveDebugFrame(const cv::Mat& frame, std::chrono::steady_clock::time_point now) {
    if (frame.empty() || now - lastDebugFrame < DEBUG_FRAME_INTERVAL) return;
    lastDebugFrame = now;

    cv::Mat annotated = frame.clone();
    tracker.drawDebug(annotated);
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
    roboControl.stopMovement();
    moving = false;
}

float FollowPerson::planHeading(float personAngle, float distance, float& clearance) {
    const float needed = std::min(distance - PERSON_MARGIN, PLAN_LOOKAHEAD);
    if (!obstacleMap.available() || needed <= 0) {
        clearance = PLAN_LOOKAHEAD;
        return personAngle;
    }

    const float x = roboControl.positionX();
    const float y = roboControl.positionY();
    const float yaw = roboControl.yaw();

    // route through everything seen in the last minutes, so the robot can use a way
    // around that it saw earlier but that is out of the LiDAR's range now
    const float targetX = x + distance * std::cos(yaw + personAngle);
    const float targetY = y + distance * std::sin(yaw + personAngle);
    std::vector<std::pair<float, float>> path;
    if (obstacleMap.planPath(x, y, targetX, targetY, PERSON_MARGIN, path) && path.size() > 1) {
        auto aim = path.back();
        for (const auto& point : path) {
            if (std::hypot(point.first - x, point.second - y) >= PATH_LOOKAHEAD) {
                aim = point;
                break;
            }
        }
        const float heading = CameraGeometry::wrapAngle(std::atan2(aim.second - y, aim.first - x) - yaw);
        const float aimDistance = std::hypot(aim.first - x, aim.second - y);
        clearance = obstacleMap.freeDistance(x, y, yaw + heading, std::max(std::min(needed, aimDistance), MIN_CLEARANCE));
        // the route already goes around everything, only nearby obstacles slow it down
        if (clearance >= std::min(needed, aimDistance)) clearance = PLAN_LOOKAHEAD;
        return heading;
    }

    // no route found (e.g. target outside the map): detour by looking straight ahead

    // try the direct heading first, then detours further and further to the side.
    // the side of the last detour is tried first so the robot doesn't zig-zag
    float bestHeading = personAngle;
    float bestFree = -1;
    for (int i = 0; i <= 2 * DETOUR_STEPS; i++) {
        const int k = (i + 1) / 2 * (i % 2 ? lastDetourSide : -lastDetourSide);
        const float heading = personAngle + k * DETOUR_STEP;
        const float free = obstacleMap.freeDistance(x, y, yaw + heading, needed);

        if (free >= needed) {
            if (k != 0) lastDetourSide = k > 0 ? 1 : -1;
            clearance = free;
            return heading;
        }
        if (free > bestFree) {
            bestFree = free;
            bestHeading = heading;
        }
    }

    // nothing is completely free: head where there is the most room
    clearance = bestFree;
    return bestHeading;
}
