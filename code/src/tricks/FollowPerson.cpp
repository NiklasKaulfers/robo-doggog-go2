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

FollowPerson::FollowPerson(core::Control& control) : roboControl(control) {
    videoClient.Init();
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

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        updateStairMode(stairSamples, levelHeight);

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
                std::cout << "[FollowPerson] target lost" << std::endl;
                roboControl.stopMovement();
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

        // z: rotate toward the person, proportional to how far off center they are.
        // with this there is no "perfect center" needed for the rotation to start
        const float middle = targetBox.xmax - (targetBox.xmax - targetBox.xmin) / 2;
        float z = std::clamp(-2.0f * (middle - 0.5f), -0.6f, +0.6f);

        if (roboControl.isStairMode()) {
            if (std::abs(roboControl.roll()) > STAIR_MAX_ROLL) {
                std::cerr << "[FollowPerson] tilted sideways on stairs, stopping" << std::endl;
                roboControl.stopMovement();
                continue;
            }

            // a nose-down body means the robot is going down the stairs
            const float maxSpeed = roboControl.pitch() > 0 ? STAIR_MAX_SPEED_DOWN : STAIR_MAX_SPEED_UP;
            x = std::clamp(x, 0.0f, maxSpeed);
            z = std::clamp(z, -STAIR_MAX_YAW, +STAIR_MAX_YAW);
        }

        if (x != 0 || z != 0)
            roboControl.moveRelative(x, z);
        else
            roboControl.stopMovement();
    }
}

void FollowPerson::updateStairMode(int& stairSamples, float& levelHeight) {
    const float pitch = std::abs(roboControl.pitch());
    const float height = roboControl.positionZ();

    if (!roboControl.isStairMode()) {
        const bool tilted = pitch > STAIR_PITCH_ENTER;
        const bool heightChanged = std::abs(height - levelHeight) > STAIR_HEIGHT_CHANGE;
        stairSamples = (tilted || heightChanged) ? stairSamples + 1 : 0;

        if (stairSamples >= STAIR_ENTER_SAMPLES) {
            std::cout << "[FollowPerson] stairs detected, switching to stair mode" << std::endl;
            if (!roboControl.setStairMode(true))
                std::cerr << "[FollowPerson] error switching to stair mode" << std::endl;
            stairSamples = 0;
        } else if (stairSamples == 0) {
            levelHeight = height;
        }
    } else {
        stairSamples = pitch < STAIR_PITCH_EXIT ? stairSamples + 1 : 0;

        if (stairSamples >= STAIR_EXIT_SAMPLES) {
            std::cout << "[FollowPerson] back on level ground" << std::endl;
            if (!roboControl.setStairMode(false))
                std::cerr << "[FollowPerson] error leaving stair mode" << std::endl;
            stairSamples = 0;
            levelHeight = height;
        }
    }
}
