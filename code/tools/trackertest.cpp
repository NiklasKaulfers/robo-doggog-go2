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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgcodecs.hpp"
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
    static cv::Rect2f headRegion(const PersonDetection& person) { return TargetTracker::headRegion(person); }
    // how fast the target is thought to walk in the world, m/s
    static float personSpeed(const TargetTracker& tracker) { return std::hypot(tracker.personVX, tracker.personVY); }
    static bool hasSignature(TargetTracker& tracker, float x) {
        auto* t = nearest(tracker, x);
        return t && !t->signature.empty();
    }
    static bool tracking(const TargetTracker& tracker, float x) {
        for (const auto& t : tracker.tracks) {
            if (t.id == tracker.targetId && std::abs(t.person.box.x - x) < 1) return true;
        }
        return false;
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


    std::cout << "13. reference photos that can't be used" << std::endl;
    {
        TargetTracker photos;
        photos.init("./res");
        const auto folder = std::filesystem::temp_directory_path() / "trackertest_photos";
        std::filesystem::remove_all(folder);
        check(!photos.loadReferencePhotos((folder / "missing").string()) && !photos.hasReferencePhotos(), "a folder that doesn't exist enrolls nothing");
        std::filesystem::create_directories(folder);
        check(!photos.loadReferencePhotos(folder.string()) && !photos.hasReferencePhotos(), "an empty folder enrolls nothing");
        cv::Mat noFace(300, 300, CV_8UC3, cv::Scalar(90, 120, 150));
        cv::circle(noFace, cv::Point(150, 150), 80, cv::Scalar(30, 30, 200), cv::FILLED);
        cv::imwrite((folder / "nobody.png").string(), noFace);
        std::ofstream((folder / "broken.jpg").string()) << "not an image";
        check(!photos.loadReferencePhotos(folder.string()) && !photos.hasReferencePhotos(), "photos without a face enroll nothing");
        // without photos the first person is still followed
        for (int i = 0; i < 6; i++) run(photos, now, {alice}, still);
        check(photos.status().visible, "so the first person is followed anyway");
        std::filesystem::remove_all(folder);
    }

    std::cout << "14. the face region of a head" << std::endl;
    {
        PersonDetection p;
        p.box = cv::Rect2f(100, 100, 200, 600);
        check(Probe::headRegion(p).empty(), "no face keypoints: no region (seen from behind)");
        auto set = [&](int k, float x, float y) { p.keypoints[k] = cv::Point2f(x, y); p.confidence[k] = 0.9f; };
        set(PersonDetection::NOSE, 200, 160);
        const float fromBox = Probe::headRegion(p).width;
        check(std::abs(fromBox - 200 * 0.35f * 1.7f) < 0.1f, "only the nose: size from the body width (" + std::to_string(fromBox) + ")");
        set(PersonDetection::LEFT_EYE, 215, 145);
        set(PersonDetection::RIGHT_EYE, 185, 145);
        const float fromEyes = Probe::headRegion(p).width;
        check(std::abs(fromEyes - 30 * 3.0f * 1.7f) < 0.1f, "with eyes: size from the eye distance (" + std::to_string(fromEyes) + ")");
        set(PersonDetection::LEFT_EAR, 230, 150);
        set(PersonDetection::RIGHT_EAR, 170, 150);
        const cv::Rect2f withEars = Probe::headRegion(p);
        check(std::abs(withEars.width - 60 * 1.4f * 1.7f) < 0.1f, "with ears: size from the ear distance (" + std::to_string(withEars.width) + ")");
        check(withEars.contains(cv::Point2f(200, 155)), "the region covers the face");
        // a tiny head is not made smaller than the detector can use
        for (int k : {1, 2, 3, 4}) p.confidence[k] = 0;
        p.box.width = 20;
        check(Probe::headRegion(p).width >= 16 * 1.7f - 0.1f, "never smaller than the minimum");
    }

    std::cout << "15. the clothes can't be seen for a moment" << std::endl;
    {
        TargetTracker hidden;
        hidden.init("./res");
        for (int i = 0; i < 12; i++) run(hidden, now, {alice}, still);
        check(hidden.status().visible && hidden.status().clothingSimilarity > 0.8f, "following them, clothes known");
        // the same box, but now a person without keypoints in a box too wide to guess the body from
        cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cv::Scalar(120, 120, 120));
        PersonDetection crouched;
        crouched.box = alice.detection().box;
        crouched.box.y += 300;
        crouched.box.height = 150;
        crouched.score = 0.8f;
        now += 100ms;
        const auto& s = hidden.update({crouched}, frame, still, now);
        check(s.visible && s.clothingSimilarity == -1, "still followed, but no clothes evidence from this frame");
        check(Probe::hasSignature(hidden, alice.x) == false, "the old signature was cleared, not kept");
    }

    std::cout << "16. the face of someone else matches the target's better" << std::endl;
    {
        TargetTracker swap;
        swap.init("./res");
        for (int i = 0; i < 12; i++) run(swap, now, {alice}, still);
        for (int i = 0; i < 3; i++) run(swap, now, {alice, bob}, still);
        check(std::abs(swap.status().target.box.x - alice.x) < 1, "following the first person");
        Probe::setFaceEvidence(swap, bob.x, 0.9f);
        const auto& s = run(swap, now, {alice, bob}, still);
        check(s.visible && std::abs(s.target.box.x - bob.x) < 1, "the target switched to the person whose face matches");
    }

    std::cout << "17. the target is dropped and a face is recognized, other clothes or not" << std::endl;
    {
        TargetTracker found;
        found.init("./res");
        for (int i = 0; i < 12; i++) run(found, now, {alice}, still);
        found.dropTarget();
        check(!found.status().visible && found.status().state == TargetTracker::State::Searching, "after dropping nobody is followed");
        Probe::setFaceEvidence(found, alice.x, 0.0f);
        for (int i = 0; i < 3; i++) run(found, now, {alice, bob}, still);
        Probe::setFaceEvidence(found, bob.x, 0.9f);
        const auto& s = run(found, now, {alice, bob}, still);
        check(s.visible && std::abs(s.target.box.x - bob.x) < 1, "the person whose face matched is followed, not the one with the right clothes");
    }

    std::cout << "17b. two faces match at once" << std::endl;
    {
        TargetTracker both;
        both.init("./res");
        Person farAlice = alice;
        farAlice.scale = 0.8f;
        for (int i = 0; i < 3; i++) run(both, now, {farAlice, bob}, still);
        both.dropTarget();
        Probe::setFaceEvidence(both, farAlice.x, 0.9f);
        Probe::setFaceEvidence(both, bob.x, 0.9f);
        const auto& s = run(both, now, {farAlice, bob}, still);
        check(s.visible && std::abs(s.target.box.x - bob.x) < 1, "with equal evidence the closer (larger) person is followed");
        both.dropTarget();
        Probe::setFaceEvidence(both, farAlice.x, 0.95f);
        const auto& t = run(both, now, {farAlice, bob}, still);
        check(t.visible && std::abs(t.target.box.x - farAlice.x) < 1, "with more evidence for the other, they are followed");
    }

    std::cout << "18. someone with other clothes appears where the target disappeared" << std::endl;
    {
        TargetTracker gap;
        gap.init("./res");
        for (int i = 0; i < 12; i++) run(gap, now, {alice}, still);
        for (int i = 0; i < 4; i++) run(gap, now, {}, still);
        check(!gap.status().visible && gap.status().state == TargetTracker::State::Lost, "the target is out of view");
        Person other = bob;
        other.x = alice.x;
        check(!run(gap, now, {other}, still).visible, "the newcomer in their place is not taken for them");
        check(!Probe::tracking(gap, other.x), "and the target is let go");
    }

    std::cout << "19. a newcomer shows up in the direction the target went" << std::endl;
    {
        TargetTracker heading;
        heading.init("./res");
        for (int i = 0; i < 12; i++) run(heading, now, {alice}, still);
        // the target track dies after 8 missed frames (0.9 s without them)
        for (int i = 0; i < 9; i++) run(heading, now, {}, still);
        check(heading.status().state == TargetTracker::State::Lost, "lost");
        // the same clothes show up for the first frame next to where they were, and far away
        Person near = alice;
        near.x = alice.x + 60;
        Person far = alice;
        far.x = 1000;
        far.scale = 0.5f;
        const auto& s = run(heading, now, {far, near}, still);
        check(s.visible && std::abs(s.target.box.x - near.x) < 1, "the one closest to where they were heading is taken");
    }

    std::cout << "20. a noisy jump of the target doesn't make them run faster than a person can" << std::endl;
    {
        TargetTracker fast;
        fast.init("./res");
        Person runner {200, alice.shirt, alice.trousers, 0.4f};
        bool followed = true;
        for (int i = 0; i < 20; i++) {
            runner.x += 60;     // 600 px/s across the image
            if (!run(fast, now, {runner}, still).visible && i > 2) followed = false;
        }
        check(followed, "followed all the way");
        const float speed = Probe::personSpeed(fast);
        check(speed > 1.4f && speed <= 1.5f + 1e-3f, "the speed is capped at 1.5 m/s (" + std::to_string(speed) + ")");
    }

    std::cout << "21. the debug drawing" << std::endl;
    {
        TargetTracker drawn;
        drawn.init("./res");
        for (int i = 0; i < 12; i++) run(drawn, now, {alice, bob}, still);
        cv::Mat image(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        drawn.drawDebug(image);
        check(cv::countNonZero(image.reshape(1)) > 1000, "people and the status are drawn");
        // the target's frame is green
        check(image.at<cv::Vec3b>(static_cast<int>(alice.top()), static_cast<int>(alice.x) + 100) == cv::Vec3b(0, 200, 0), "the target's box is green");
        check(image.at<cv::Vec3b>(static_cast<int>(bob.top()), static_cast<int>(bob.x) + 100) == cv::Vec3b(160, 160, 160), "the others' is gray");
        // a face that surely isn't the person's turns a box red
        Probe::setFaceEvidence(drawn, bob.x, -0.8f);
        cv::Mat red(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        drawn.drawDebug(red);
        check(red.at<cv::Vec3b>(static_cast<int>(bob.top()), static_cast<int>(bob.x) + 100) == cv::Vec3b(0, 0, 220), "a rejected face is red");
        // and while lost
        for (int i = 0; i < 4; i++) run(drawn, now, {bob}, still);
        cv::Mat lost(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        drawn.drawDebug(lost);
        check(cv::countNonZero(lost.reshape(1)) > 0, "drawing while lost works");
        cv::Mat nothing;
        drawn.drawDebug(nothing);
        check(nothing.empty(), "an empty image is left alone");
    }

    std::cout << (failures ? "\nFAILED" : "\nall checks passed") << std::endl;
    return failures ? 1 : 0;
}
