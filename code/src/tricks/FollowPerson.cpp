#include "FollowPerson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include "../ml/movenet/Multipose.h"

using namespace robo::sdk::examples;
using namespace robo::sdk::ml;

// The width of a standing person in normalized image coordinates (0..1) when they
// stand approximately 0.5m away from the Go2's front camera. Movenet's boxes scale
// roughly inversely with the distance to the camera, so comparing the detected box
// width against this value gives an estimate of the current distance between the
// robot and the person. It is calibrated for a ~0.4m wide person (shoulders) in front
// of the Go2's 120° FOV camera and can be tuned per installation.
static constexpr float BOX_WIDTH_AT_0_5M = 0.35f;
static constexpr float TARGET_DISTANCE = 0.5f;
static constexpr float PERSON_CENTER_TOLERANCE = 0.15f;

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

// horizontal field of view of the front camera, maps image position to an angle
static constexpr float CAMERA_FOV = 2.09f;          // 120 degrees
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

FollowPerson::FollowPerson(core::Control& control) : roboControl(control) {
    videoClient.Init();
    obstacleMap.init();
}

void FollowPerson::run() {
    runWithMultipose();
}

void FollowPerson::runWithMultipose() {
    movenet::Multipose detector;
    if (!detector.init()) return;

    // the first person we saw, which is tracked until it is lost
    bool targetFound = false;
    movenet::Box targetBox {0, 0, 0, 0, 0};
    int countNotFound = 0;

    int stairSamples = 0;
    float levelHeight = roboControl.positionZ();

    while (!core::Control::stopRequested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        updateStairMode(stairSamples, levelHeight);

        // the remote controller always wins: don't send anything while it is used
        if (roboControl.controllerActive()) {
            roboControl.releaseControl();
            moving = false;
            continue;
        }

        videoClient.GetImageSample(imgData);
        if (!detector.run(imgData)) {
            std::cerr << "[FollowPerson] error in detector.run()" << std::endl;
            continue;
        }

        // recognize the person: same logic as TurnToPerson, the detection with
        // the highest confidence is used
        movenet::Box mark {0, 0, 0, 0, 0};
        int detectionIdx = 0;
        for (int d = 0; d < detector.modelDetectionCount(); d++) {
            const auto box = detector.getBox(d);
            if (box.confidence > mark.confidence) {
                mark = box;
                detectionIdx = d;
            }
        }

        // first person seen -> make it the target we follow
        if (!targetFound) {
            if (mark.confidence < 0.15f) continue;
            targetFound = true;
            targetBox = mark;
            countNotFound = 0;
            std::cout << "[FollowPerson] found person, following" << std::endl;
        } else {
            // keep tracking the same person: the center of a detection must stay
            // close to the one we already follow
            const float cx = mark.xmax - (mark.xmax - mark.xmin) / 2;
            const float cy = mark.ymax - (mark.ymax - mark.ymin) / 2;
            const float tx = targetBox.xmax - (targetBox.xmax - targetBox.xmin) / 2;
            const float ty = targetBox.ymax - (targetBox.ymax - targetBox.ymin) / 2;

            if ((mark.confidence >= 0.15f) &&
                std::sqrt((cx - tx) * (cx - tx) + (cy - ty) * (cy - ty)) < PERSON_CENTER_TOLERANCE) {
                targetBox = mark;
                countNotFound = 0;
            } else if (++countNotFound > 8) {
                if (moving) std::cout << "[FollowPerson] target lost" << std::endl;
                stop();
                continue;
            }
        }

        // estimate the distance from the robot to the person: the apparent width of a
        // standing person scales roughly inversely with their distance to the camera,
        // which gives a much more sensitive measurement than the box height here
        const float boxWidth = targetBox.xmax - targetBox.xmin;
        const float distance = TARGET_DISTANCE * BOX_WIDTH_AT_0_5M / boxWidth;

        // proportional control so the motion is smooth instead of walking straight and
        // suddenly turning once the person leaves a band around the image center
        // (x: move toward the person if too far, away if too close)
        float x = std::clamp(0.5f * (distance - TARGET_DISTANCE), -0.6f, +0.6f);

        // direction of the person relative to the robot's heading
        const float middle = targetBox.xmax - (targetBox.xmax - targetBox.xmin) / 2;
        const float personAngle = -(middle - 0.5f) * CAMERA_FOV;

        // when walking toward the person, take a detour around obstacles in the way
        float heading = personAngle;
        if (x > 0) {
            float clearance = 0;
            heading = planHeading(personAngle, std::min(distance - PERSON_MARGIN, PLAN_LOOKAHEAD), clearance);
            // slow down in front of obstacles and only turn if nothing is free
            x = std::min(x, std::max(0.0f, 0.5f * (clearance - MIN_CLEARANCE)));
        }

        // z: rotate toward the chosen heading, proportional to how far off it is.
        // with this there is no "perfect center" needed for the rotation to start
        float z = std::clamp(TURN_GAIN * heading, -0.6f, +0.6f);

        if (stairMode) {
            if (std::abs(roboControl.roll()) > STAIR_MAX_ROLL) {
                if (moving) std::cerr << "[FollowPerson] tilted sideways on stairs, stopping" << std::endl;
                stop();
                continue;
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

float FollowPerson::planHeading(float personAngle, float needed, float& clearance) {
    if (!obstacleMap.available() || needed <= 0) {
        clearance = PLAN_LOOKAHEAD;
        return personAngle;
    }

    const float x = roboControl.positionX();
    const float y = roboControl.positionY();
    const float yaw = roboControl.yaw();

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
