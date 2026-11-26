#include "TurnToPerson.h"

#include <chrono>
#include <thread>
#include "../ml/movenet/Singlepose.h"
#include "../ml/movenet/Multipose.h"
#include "../logging/Logger.h"

namespace logging = robo::sdk::logging;

using namespace robo::sdk::examples;
using namespace robo::sdk::ml;

TurnToPerson::TurnToPerson(core::Control& control) : roboControl(control) {
    videoClient.Init();
}

void TurnToPerson::run() {
    //runWithSinglepose();
    runWithMultipose();
}

void TurnToPerson::runWithSinglepose() {
    movenet::Singlepose detector;
    if (!detector.init()) return;

    while (!core::Control::stopRequested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        videoClient.GetImageSample(imgData);
        if (!detector.run(imgData)) {
            logging::error("TurnToPerson", "error in detector.run()");
            continue;
        }

        const auto mark = detector.getKeypoint(0, movenet::Movenet::KEYPOINT_LEFT_KNEE);

        if (mark.confidence < 0.15) {
            countNotFound++;
            if (countNotFound > 8) {
                logging::info("TurnToPerson", "target lost");
                roboControl.stopMovement();
            }
            continue;
        }
        countNotFound = 0;

        if (mark.y > 0.55)
            roboControl.rotateRight(0.6f);
        else if (mark.y < 0.45)
            roboControl.rotateLeft(0.6f);
        else
            roboControl.stopMovement();
    }
}

void TurnToPerson::runWithMultipose() {
    movenet::Multipose detector;
    if (!detector.init()) return;

    while (!core::Control::stopRequested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        videoClient.GetImageSample(imgData);
        if (!detector.run(imgData)) {
            logging::error("TurnToPerson", "error in detector.run()");
            continue;
        }

        movenet::Box mark {0, 0, 0, 0, 0};
        int detectionIdx = 0;
        for (int d = 0; d < detector.modelDetectionCount(); d++) {
            const auto box = detector.getBox(d);
            if (box.confidence > mark.confidence) {
                mark = box;
                detectionIdx = d;
            }
        }
        logging::info("TurnToPerson", logging::text("using detection ", detectionIdx));

        if (mark.confidence < 0.15) {
            countNotFound++;
            if (countNotFound > 8) {
                logging::info("TurnToPerson", "target lost");
                roboControl.stopMovement();
            }
            continue;
        }
        countNotFound = 0;

        float middle = mark.xmax - (mark.xmax - mark.xmin) / 2;
        if (middle > 0.55)
            roboControl.rotateRight(0.6f);
        else if (middle < 0.45)
            roboControl.rotateLeft(0.6f);
        else
            roboControl.stopMovement();
    }
}
