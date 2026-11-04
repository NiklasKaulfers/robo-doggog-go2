#include "SitStandApproach.h"

#include <chrono>
#include <thread>
#include "../ml/movenet/Singlepose.h"

using namespace robo::sdk::examples;
using namespace robo::sdk::ml;

SitStandApproach::SitStandApproach(core::Control& control) : roboControl(control) {
    videoClient.Init();
}

void SitStandApproach::run() {
    runWithSinglepose();
}

void SitStandApproach::runWithSinglepose() {
    movenet::Singlepose detector;
    if (!detector.init()) return;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        videoClient.GetImageSample(imgData);
        if (!detector.run(imgData)) {
            std::cerr << "[TurnToPerson] error in detector.run()" << std::endl;
            continue;
        }

        if (checkGestureSit(detector)) continue;
        if (checkGestureStand(detector)) continue;
        if (checkGestureApproach(detector)) continue;
    }
}

bool SitStandApproach::checkGestureSit(const movenet::Movenet &detector) {
    const auto mark1 = detector.getKeypoint(0, movenet::Movenet::KEYPOINT_RIGHT_WRIST);
    const auto mark2 = detector.getKeypoint(0, movenet::Movenet::KEYPOINT_RIGHT_ANKLE);

    if (mark1.confidence < 0.15 || mark2.confidence < 0.15)
        return false;

    const float distance = std::sqrt(std::abs(mark1.x - mark2.x) + std::abs(mark1.y - mark2.y));
    if (distance > 0.3)
        return false;

    roboControl.sitdown();
    return true;
}

bool SitStandApproach::checkGestureStand(const movenet::Movenet &detector) {
    const auto mark1 = detector.getKeypoint(0, movenet::Movenet::KEYPOINT_RIGHT_WRIST);
    const auto mark2 = detector.getKeypoint(0, movenet::Movenet::KEYPOINT_RIGHT_SHOULDER);

    if (mark1.confidence < 0.15 || mark2.confidence < 0.15)
        return false;

    const float distance = std::sqrt(std::abs(mark1.x - mark2.x) + std::abs(mark1.y - mark2.y));
    if (distance > 0.3)
        return false;

    roboControl.standup();
    return true;
}

bool SitStandApproach::checkGestureApproach(const movenet::Movenet &detector) {
    const auto mark1 = detector.getKeypoint(0, movenet::Movenet::KEYPOINT_RIGHT_WRIST);
    const auto mark2 = detector.getKeypoint(0, movenet::Movenet::KEYPOINT_LEFT_WRIST);

    if (mark1.confidence < 0.15 || mark2.confidence < 0.15)
        return false;

    const float distance = std::sqrt(std::abs(mark1.x - mark2.x) + std::abs(mark1.y - mark2.y));
    if (distance > 0.3)
        return false;

    roboControl.moveForwardBackward(0.5f);
    return true;
}
