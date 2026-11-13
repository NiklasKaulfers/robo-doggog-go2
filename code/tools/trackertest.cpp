// Self-test of the TargetTracker with drawn people instead of camera frames: checks that
// the first person is locked on, their clothes learned, that a second person with other
// clothes is neither followed nor taken for them, that they are found again by their
// clothes after being out of view, and that a swap (someone else standing where they
// were) is noticed. Runs without the robot:
//
//   cd code && ./build/trackertest
//
// (needs ./res with the face models, like the app). Exit code 0 when all checks pass.

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "../src/ml/PersonDetection.h"
#include "../src/tracking/TargetTracker.h"

using namespace robo::sdk::ml;
using namespace robo::sdk::tracking;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok    " : "  FAIL  ") << what << std::endl;
    if (!condition) failures++;
}

// a standing person seen from behind (no face keypoints), x is the left edge of their body
struct Person {
    float x;
    cv::Scalar shirt, trousers;

    PersonDetection detection() const {
        PersonDetection p;
        p.box = cv::Rect2f(x, 100, 200, 600);
        p.score = 0.8f;
        auto set = [&](int k, float px, float py) {
            p.keypoints[k] = cv::Point2f(x + px, py);
            p.confidence[k] = 0.9f;
        };
        set(PersonDetection::LEFT_SHOULDER, 40, 220);
        set(PersonDetection::RIGHT_SHOULDER, 160, 220);
        set(PersonDetection::LEFT_HIP, 60, 420);
        set(PersonDetection::RIGHT_HIP, 140, 420);
        set(PersonDetection::LEFT_KNEE, 65, 560);
        set(PersonDetection::RIGHT_KNEE, 135, 560);
        set(PersonDetection::LEFT_ANKLE, 65, 690);
        set(PersonDetection::RIGHT_ANKLE, 135, 690);
        return p;
    }

    void draw(cv::Mat& frame) const {
        cv::rectangle(frame, cv::Rect(x + 30, 200, 140, 230), shirt, cv::FILLED);
        cv::rectangle(frame, cv::Rect(x + 40, 430, 120, 270), trousers, cv::FILLED);
    }
};

int main() {
    TargetTracker tracker;
    if (!tracker.init("./res")) {
        std::cerr << "run from the code folder, ./res with the face models is needed" << std::endl;
        return 1;
    }

    const Person alice {500, cv::Scalar(40, 40, 220), cv::Scalar(160, 60, 30)};   // red shirt, blue jeans
    const Person bob {900, cv::Scalar(60, 180, 60), cv::Scalar(30, 30, 30)};      // green shirt, black trousers
    const RobotPose pose;
    auto now = std::chrono::steady_clock::now();

    auto step = [&](const std::vector<Person>& people) -> const TargetTracker::Status& {
        now += std::chrono::milliseconds(100);
        cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(120, 120, 120));
        std::vector<PersonDetection> detections;
        for (const auto& person : people) {
            person.draw(frame);
            detections.push_back(person.detection());
        }
        return tracker.update(detections, frame, pose, now);
    };

    std::cout << "1. nobody in view" << std::endl;
    for (int i = 0; i < 3; i++) step({});
    check(tracker.status().state == TargetTracker::State::Searching, "searching while nobody is there");

    std::cout << "2. the first person shows up and stays" << std::endl;
    for (int i = 0; i < 15; i++) step({alice});
    check(tracker.status().visible, "following them");
    check(std::abs(tracker.status().target.box.x - alice.x) < 1, "the target is the first person");

    std::cout << "3. a second person with other clothes comes into view" << std::endl;
    for (int i = 0; i < 5; i++) step({alice, bob});
    check(tracker.status().visible && std::abs(tracker.status().target.box.x - alice.x) < 1, "still following the first person");
    check(tracker.status().clothingSimilarity > 0.8f, "their clothes look like the learned ones (" + std::to_string(tracker.status().clothingSimilarity) + ")");

    std::cout << "4. the first person leaves the view, the second stays" << std::endl;
    for (int i = 0; i < 12; i++) step({bob});
    check(!tracker.status().visible, "the target is not in view");
    check(tracker.status().state == TargetTracker::State::Lost, "state is lost");

    std::cout << "5. the first person comes back somewhere else" << std::endl;
    Person aliceMoved = alice;
    aliceMoved.x = 100;
    for (int i = 0; i < 4; i++) step({aliceMoved, bob});
    check(tracker.status().visible && std::abs(tracker.status().target.box.x - aliceMoved.x) < 1, "found again by their clothes, not the other person");

    std::cout << "6. someone with other clothes takes their place while they are hidden" << std::endl;
    Person impostor = bob;
    impostor.x = aliceMoved.x;
    for (int i = 0; i < 8; i++) step({impostor});
    check(!tracker.status().visible, "the swap was noticed and the target dropped");

    std::cout << "7. the first person is back in view" << std::endl;
    for (int i = 0; i < 4; i++) step({aliceMoved, bob});
    check(tracker.status().visible && std::abs(tracker.status().target.box.x - aliceMoved.x) < 1, "followed again");

    std::cout << (failures ? "\nFAILED" : "\nall checks passed") << std::endl;
    return failures ? 1 : 0;
}
