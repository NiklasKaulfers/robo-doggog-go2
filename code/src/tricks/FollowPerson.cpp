#include "FollowPerson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include "../ml/movenet/Multipose.h"

using namespace robo::sdk::examples;
using namespace robo::sdk::ml;

// The height of a standing person in normalized image coordinates (0..1) when
// they stand approximately 0.5m away from the Go2's front camera. Movenet's
// boxes scale roughly inversely with the distance to the camera, so comparing
// the detected box height against this value gives an estimate of the current
// distance between the robot and the person. This constant can be tuned for
// a specific installation (camera position / person height).
static constexpr float BOX_HEIGHT_AT_0_5M = 0.75f;
static constexpr float TARGET_DISTANCE = 0.5f;
static constexpr float PERSON_CENTER_TOLERANCE = 0.15f;

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

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

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

        // keep the distance to the person around TARGET_DISTANCE (0.5m)
        const float boxHeight = targetBox.ymax - targetBox.ymin;
        const float distance = TARGET_DISTANCE * BOX_HEIGHT_AT_0_5M / boxHeight;

        float x = 0.0f;
        if (distance > TARGET_DISTANCE) {
            // too far away -> walk towards the person
            x = std::min(0.6f, 0.5f * (distance - TARGET_DISTANCE));
        } else if (distance < TARGET_DISTANCE) {
            // too close -> walk away from the person
            x = -std::min(0.6f, 0.5f * (TARGET_DISTANCE - distance));
        }

        // keep the person centered in front of the robot
        const float middle = targetBox.xmax - (targetBox.xmax - targetBox.xmin) / 2;
        float z = 0.0f;
        if (middle > 0.55f) {
            z = -0.6f;
        } else if (middle < 0.45f) {
            z = +0.6f;
        }

        // a single Move is used so that translation and rotation are applied at
        // the same time (successive Move calls would override each other)
        if (x != 0 || z != 0)
            roboControl.moveRelative(x, z);
        else
            roboControl.stopMovement();
    }
}
