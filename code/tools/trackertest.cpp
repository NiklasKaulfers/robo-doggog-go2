// Self-test of the TargetTracker with drawn people instead of camera frames: checks that
// the first person is locked on, their clothes learned, that a second person with other
// clothes is neither followed nor taken for them, that they are found again by their
// clothes after being out of view, and that a swap (someone else standing where they
// were) is noticed. Then: the target is kept while the robot turns, a face mismatch
// drops them for good, stale face observations don't count again, and a long pause
// between frames neither extrapolates boxes nor hands the target to someone else.
// Faces can't be drawn (YuNet finds none), so face evidence is set directly. Runs
// without the robot:
//
//   cd code && ./build/trackertest
//
// (needs ./res with the face models, like the app). Exit code 0 when all checks pass.

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "../src/ml/PersonDetection.h"
#include "../src/tracking/CameraGeometry.h"
#include "../src/tracking/TargetTracker.h"

using namespace robo::sdk::ml;
using namespace robo::sdk::tracking;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok    " : "  FAIL  ") << what << std::endl;
    if (!condition) failures++;
}

// a standing person seen from behind (no face keypoints), x is the left edge of their body.
// scale < 1 makes them smaller (further away), standing on the same floor line; with
// head they face the camera (nose and eyes, but no face is drawn, so none is found)
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
            return cv::Rect(cv::Point(x + px * scale, top() + (py - 100) * scale),
                            cv::Size(w * scale, h * scale));
        };
        cv::rectangle(frame, rect(30, 200, 140, 230), shirt, cv::FILLED);
        cv::rectangle(frame, rect(40, 430, 120, 270), trousers, cv::FILLED);
    }
};

namespace robo::sdk::tracking {
// faces can't be drawn, so the face evidence is looked at and set directly
struct TargetTrackerProbe {
    static TargetTracker::Track* nearest(TargetTracker& tracker, float x) {
        TargetTracker::Track* best = nullptr;
        for (auto& t : tracker.tracks) {
            if (!best || std::abs(t.person.box.x - x) < std::abs(best->person.box.x - x)) best = &t;
        }
        return best;
    }
    static void setFaceEvidence(TargetTracker& tracker, float x, float evidence) {
        if (auto* t = nearest(tracker, x)) t->faceEvidence = evidence;
    }
    static float faceEvidence(TargetTracker& tracker, float x) {
        auto* t = nearest(tracker, x);
        return t ? t->faceEvidence : 0;
    }
    // as if a clear face of someone else had been seen on them in the last frame
    static void setMismatchFace(TargetTracker& tracker, float x) {
        auto* t = nearest(tracker, x);
        if (!t) return;
        t->face = FaceRecognizer::Observation();
        t->face.found = true;
        t->face.size = 60;
        t->face.similarity = 0.05f;
        t->face.verdict = FaceRecognizer::Verdict::Mismatch;
    }
    static std::vector<cv::Rect2f> boxes(const TargetTracker& tracker) {
        std::vector<cv::Rect2f> result;
        for (const auto& t : tracker.tracks) result.push_back(t.person.box);
        return result;
    }
};
}

using Probe = robo::sdk::tracking::TargetTrackerProbe;

static constexpr int FRAME_WIDTH = 1280, FRAME_HEIGHT = 720;

// one frame with these people for the tracker, dt after the last one
static const TargetTracker::Status& run(TargetTracker& tracker, std::chrono::steady_clock::time_point& now,
                                        const std::vector<Person>& people, const RobotPose& pose,
                                        std::chrono::milliseconds dt = std::chrono::milliseconds(100)) {
    now += dt;
    cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cv::Scalar(120, 120, 120));
    std::vector<PersonDetection> detections;
    for (const auto& person : people) {
        person.draw(frame);
        detections.push_back(person.detection());
    }
    return tracker.update(detections, frame, pose, now);
}

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
        return run(tracker, now, people, pose);
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

    // the scenarios below each start with a fresh tracker
    using namespace std::chrono_literals;
    const RobotPose still;

    std::cout << "8. the robot turns toward a person standing still (far away, frames every 200 ms)" << std::endl;
    {
        TargetTracker turner;
        turner.init("./res");
        RobotPose turning;
        // they stand at a fixed direction in the world, this far left of the start heading
        const float bearing = 0.64f;
        auto seenAt = [&](float yaw) {
            Person p {0, alice.shirt, alice.trousers, 0.3f};
            const float center = (0.5f - (bearing - yaw) / CameraGeometry::FOV) * FRAME_WIDTH;
            p.x = center - p.width() / 2;
            return p;
        };
        for (int i = 0; i < 12; i++) run(turner, now, {seenAt(turning.yaw)}, turning, 200ms);
        check(turner.status().visible, "following them before turning");
        bool kept = true;
        for (int i = 0; i < 10; i++) {
            // 0.6 rad/s to the left: they move right in the image by ~74 px per frame
            turning.yaw += 0.12f;
            const Person p = seenAt(turning.yaw);
            const auto& s = run(turner, now, {p}, turning, 200ms);
            if (!s.visible || std::abs(s.target.box.x - p.x) > 1) kept = false;
        }
        check(kept, "kept in every frame while the robot turns");
    }

    std::cout << "9. the face of the target turns out to be someone else's (same clothes)" << std::endl;
    {
        TargetTracker faced;
        faced.init("./res");
        for (int i = 0; i < 15; i++) run(faced, now, {alice}, still);
        check(faced.status().visible, "following them");
        // as after clear face mismatches
        Probe::setFaceEvidence(faced, alice.x, -0.6f);
        bool followedAgain = false;
        for (int i = 0; i < 5; i++) {
            if (run(faced, now, {alice}, still).visible) followedAgain = true;
        }
        check(!followedAgain, "dropped, and not picked again by their clothes");
        check(faced.status().state == TargetTracker::State::Lost, "state is lost");
        check(Probe::faceEvidence(faced, alice.x) < 0, "the evidence against their face is kept");
    }

    std::cout << "10. five people facing the camera: a face is only counted in the frame it was seen" << std::endl;
    {
        TargetTracker crowded;
        crowded.init("./res");
        const cv::Scalar shirts[] = {{40, 40, 220}, {60, 180, 60}, {200, 80, 40}, {40, 200, 220}, {180, 40, 180}};
        std::vector<Person> crowd;
        for (int i = 0; i < 5; i++) {
            // the last one a bit smaller, so they are the last whose face is checked
            crowd.push_back({20.0f + 250 * i, shirts[i], cv::Scalar(30, 30, 30), i == 4 ? 0.9f : 1.0f, true});
        }
        for (int i = 0; i < 3; i++) run(crowded, now, crowd, still);
        Probe::setMismatchFace(crowded, crowd[4].x);
        for (int i = 0; i < 3; i++) run(crowded, now, crowd, still);
        const float evidence = Probe::faceEvidence(crowded, crowd[4].x);
        check(evidence > -0.5f, "one old mismatch doesn't add up to a rejection (evidence " + std::to_string(evidence) + ")");
    }

    std::cout << "11. no frames for a minute after following someone walking" << std::endl;
    {
        TargetTracker paused;
        paused.init("./res");
        Person walker = alice;
        walker.x = 200;
        for (int i = 0; i < 15; i++) {
            walker.x += 20;     // 200 px/s
            run(paused, now, {walker}, still);
        }
        check(paused.status().visible, "following them");
        run(paused, now, {}, still, 60s);
        bool sane = true;
        for (const auto& box : Probe::boxes(paused)) {
            if (box.x < -FRAME_WIDTH || box.x > 2 * FRAME_WIDTH) sane = false;
        }
        check(sane, "no box extrapolated far out of the image");
    }

    std::cout << "12. after a minute without frames (the robot turned around), someone else stands where the target was" << std::endl;
    {
        TargetTracker paused;
        paused.init("./res");
        RobotPose pose12;
        for (int i = 0; i < 15; i++) run(paused, now, {alice}, pose12);
        check(paused.status().visible, "following them");
        pose12.yaw += 2 * static_cast<float>(CV_PI) + 0.05f;
        Person other = bob;
        other.x = alice.x;
        check(!run(paused, now, {other}, pose12, 60s).visible, "the other person is not taken for the target");
        for (int i = 0; i < 3; i++) run(paused, now, {aliceMoved, other}, pose12);
        check(paused.status().visible && std::abs(paused.status().target.box.x - aliceMoved.x) < 1, "the target is found again by their clothes");
    }

    std::cout << (failures ? "\nFAILED" : "\nall checks passed") << std::endl;
    return failures ? 1 : 0;
}
