#ifndef FOLLOW_TESTS_DRAWNPERSON_H
#define FOLLOW_TESTS_DRAWNPERSON_H

// People drawn into a frame instead of camera images: a shirt and trousers rectangle
// plus the pose detection that goes with them. The TargetTracker takes the clothes from
// the drawing at the keypoints, exactly like from a camera frame. Faces can't be drawn
// (the face detector finds none), face evidence is set through TargetTrackerProbe.

#include <chrono>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "ml/PersonDetection.h"
#include "tracking/TargetTracker.h"

namespace testutil {

using robo::sdk::ml::PersonDetection;

constexpr int FRAME_WIDTH = 1280, FRAME_HEIGHT = 720;
// background of the drawn frames (BGR)
const cv::Scalar BACKGROUND(120, 120, 120);

// a standing person seen from behind (no face keypoints), x is the left edge of their
// body. scale < 1 makes them smaller (further away), standing on the same floor line;
// with head they face the camera (nose and eyes, but no face is drawn, so none is found)
struct Person {
    float x;
    cv::Scalar shirt, trousers;
    float scale = 1;
    bool head = false;

    // at scale 1 the body is 200 x 600 px with its top at y = 100
    float top() const { return 700 - 600 * scale; }
    float width() const { return 200 * scale; }

    PersonDetection detection() const {
        PersonDetection p;
        p.box = cv::Rect2f(x, top(), 200 * scale, 600 * scale);
        p.score = 0.8f;
        auto set = [&](int k, float px, float py) {
            p.keypoints[k] = cv::Point2f(x + px * scale, top() + (py - 100) * scale);
            p.confidence[k] = 0.9f;
        };
        if (head) {
            set(PersonDetection::NOSE, 100, 160);
            set(PersonDetection::LEFT_EYE, 115, 145);
            set(PersonDetection::RIGHT_EYE, 85, 145);
        }
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
        auto rect = [&](float px, float py, float w, float h) {
            return cv::Rect(cv::Point(x + px * scale, top() + (py - 100) * scale), cv::Size(w * scale, h * scale));
        };
        cv::rectangle(frame, rect(30, 200, 140, 230), shirt, cv::FILLED);
        cv::rectangle(frame, rect(40, 430, 120, 270), trousers, cv::FILLED);
    }
};

// some clothes that are easy to tell apart (BGR)
const cv::Scalar RED(40, 40, 220), GREEN(60, 180, 60), BLUE(200, 80, 40), YELLOW(40, 200, 220), PURPLE(180, 40, 180);
const cv::Scalar JEANS(160, 60, 30), BLACK(30, 30, 30);

// a frame with these people drawn in
inline cv::Mat drawFrame(const std::vector<Person>& people) {
    cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, BACKGROUND);
    for (const auto& person : people) person.draw(frame);
    return frame;
}

// feeds one frame with these people to the tracker, dt after the last one (now is advanced)
inline const robo::sdk::tracking::TargetTracker::Status& runFrame(
        robo::sdk::tracking::TargetTracker& tracker, std::chrono::steady_clock::time_point& now,
        const std::vector<Person>& people, const robo::sdk::tracking::RobotPose& pose = {},
        std::chrono::milliseconds dt = std::chrono::milliseconds(100)) {
    now += dt;
    const cv::Mat frame = drawFrame(people);
    std::vector<PersonDetection> detections;
    for (const auto& person : people) detections.push_back(person.detection());
    return tracker.update(detections, frame, pose, now);
}
}

#endif //FOLLOW_TESTS_DRAWNPERSON_H
