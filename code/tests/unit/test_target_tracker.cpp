// TargetTracker: who is followed, with drawn people instead of camera frames. The face
// models are not loaded (init() is not called), so no face is ever found; face evidence
// is set through TargetTrackerProbe where a scenario needs it.
#include <cmath>
#include <vector>
#include "doctest/doctest.h"
#include "tracking/CameraGeometry.h"
#include "DrawnPerson.h"
#include "TargetTrackerProbe.h"

using namespace robo::sdk::tracking;
using namespace testutil;
using Probe = TargetTrackerProbe;
using State = TargetTracker::State;
using namespace std::chrono_literals;

namespace {

const Person ALICE {500, RED, JEANS};
const Person BOB {900, GREEN, BLACK};

// a tracker with its own clock and a robot that stands still unless the test moves it
struct Rig {
    TargetTracker tracker;
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    RobotPose pose;

    const TargetTracker::Status& step(const std::vector<Person>& people,
                                      std::chrono::milliseconds dt = 100ms) {
        return runFrame(tracker, now, people, pose, dt);
    }
    void steps(int count, const std::vector<Person>& people) {
        for (int i = 0; i < count; i++) step(people);
    }
    bool following(float x) const {
        const auto& s = tracker.status();
        return s.visible && std::abs(s.target.box.x - x) < 1;
    }
};

// a face feature for the gallery, as if a reference photo had been loaded
cv::Mat referenceFeature() {
    const std::vector<float> values(128, 0.1f);
    return cv::Mat(values).clone();
}

// where the box is in the world with the robot at the origin looking along +x
cv::Point2f worldPosition(const cv::Rect2f& box) {
    const float distance = CameraGeometry::distance(box, FRAME_WIDTH);
    const float angle = CameraGeometry::angle(box, FRAME_WIDTH);
    return {distance * std::cos(angle), distance * std::sin(angle)};
}
}

TEST_CASE("searching while nobody is in view") {
    TargetTracker tracker;
    auto now = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; i++) runFrame(tracker, now, {});
    CHECK(tracker.status().state == TargetTracker::State::Searching);
}

// trackertest scenarios 1 to 7: one story on one tracker
TEST_CASE("first person is locked on and kept apart from others across leaving and returning") {
    Rig r;
    const Person alice = ALICE, bob = BOB;

    // scenario 1: nobody in view
    r.steps(3, {});
    CHECK(r.tracker.status().state == State::Searching);

    // scenario 2: the first person shows up and stays
    r.steps(15, {alice});
    CHECK(r.tracker.status().visible);
    CHECK(std::abs(r.tracker.status().target.box.x - alice.x) < 1);

    // scenario 3: a second person with other clothes comes into view
    r.steps(5, {alice, bob});
    CHECK(r.following(alice.x));
    CHECK(r.tracker.status().clothingSimilarity > 0.8f);

    // scenario 4: the first person leaves, the second stays
    r.steps(12, {bob});
    CHECK_FALSE(r.tracker.status().visible);
    CHECK(r.tracker.status().state == State::Lost);

    // scenario 5: the first person comes back somewhere else
    Person aliceMoved = alice;
    aliceMoved.x = 100;
    r.steps(4, {aliceMoved, bob});
    CHECK(r.following(aliceMoved.x));

    // scenario 6: someone with other clothes takes their place while they are hidden
    Person impostor = bob;
    impostor.x = aliceMoved.x;
    r.steps(8, {impostor});
    CHECK_FALSE(r.tracker.status().visible);

    // scenario 7: the first person is back in view
    r.steps(4, {aliceMoved, bob});
    CHECK(r.following(aliceMoved.x));
}

TEST_CASE("target is kept in every frame while the robot turns toward a person standing still") {
    // trackertest scenario 8: far away, frames every 200 ms
    Rig r;
    RobotPose turning;
    // they stand at a fixed direction in the world, this far left of the start heading
    const float bearing = 0.64f;
    auto seenAt = [&](float yaw) {
        Person p {0, RED, JEANS, 0.3f};
        const float center = (0.5f - (bearing - yaw) / CameraGeometry::FOV) * FRAME_WIDTH;
        p.x = center - p.width() / 2;
        return p;
    };
    for (int i = 0; i < 12; i++) runFrame(r.tracker, r.now, {seenAt(turning.yaw)}, turning, 200ms);
    CHECK(r.tracker.status().visible);
    bool kept = true;
    for (int i = 0; i < 10; i++) {
        // 0.6 rad/s to the left: they move right in the image by ~74 px per frame
        turning.yaw += 0.12f;
        const Person p = seenAt(turning.yaw);
        const auto& s = runFrame(r.tracker, r.now, {p}, turning, 200ms);
        if (!s.visible || std::abs(s.target.box.x - p.x) > 1) kept = false;
    }
    CHECK(kept);
}

TEST_CASE("target whose face turns out to be someone else's is dropped and not picked again by clothes") {
    // trackertest scenario 9
    Rig r;
    r.steps(15, {ALICE});
    CHECK(r.tracker.status().visible);
    // as after clear face mismatches
    Probe::setFaceEvidence(r.tracker, ALICE.x, -0.6f);
    bool followedAgain = false;
    for (int i = 0; i < 5; i++) {
        if (r.step({ALICE}).visible) followedAgain = true;
    }
    CHECK_FALSE(followedAgain);
    CHECK(r.tracker.status().state == State::Lost);
    CHECK(Probe::faceEvidence(r.tracker, ALICE.x) < 0);
}

TEST_CASE("an old face mismatch seen on a crowded frame does not count again later") {
    // trackertest scenario 10: five people facing the camera
    Rig r;
    const cv::Scalar shirts[] = {RED, GREEN, BLUE, YELLOW, PURPLE};
    std::vector<Person> crowd;
    for (int i = 0; i < 5; i++) {
        // the last one a bit smaller, so they are the last whose face is checked
        crowd.push_back({20.0f + 250 * i, shirts[i], BLACK, i == 4 ? 0.9f : 1.0f, true});
    }
    r.steps(3, crowd);
    Probe::setMismatchFace(r.tracker, crowd[4].x);
    r.steps(3, crowd);
    CHECK(Probe::faceEvidence(r.tracker, crowd[4].x) > -0.5f);
}

TEST_CASE("no frames for a minute after following someone walking leaves no boxes far outside the image") {
    // trackertest scenario 11
    Rig r;
    Person walker = ALICE;
    walker.x = 200;
    for (int i = 0; i < 15; i++) {
        walker.x += 20;     // 200 px/s
        r.step({walker});
    }
    CHECK(r.tracker.status().visible);
    r.step({}, 60s);
    bool sane = true;
    for (const auto& box : Probe::boxes(r.tracker)) {
        if (box.x < -FRAME_WIDTH || box.x > 2 * FRAME_WIDTH) sane = false;
    }
    CHECK(sane);
}

TEST_CASE("after a minute without frames someone else where the target was is not taken for them") {
    // trackertest scenario 12: the robot turned around meanwhile
    Rig r;
    Person aliceMoved = ALICE;
    aliceMoved.x = 100;
    r.steps(15, {ALICE});
    CHECK(r.tracker.status().visible);
    r.pose.yaw += 2 * static_cast<float>(CV_PI) + 0.05f;
    Person other = BOB;
    other.x = ALICE.x;
    CHECK_FALSE(r.step({other}, 60s).visible);
    r.steps(3, {aliceMoved, other});
    CHECK(r.following(aliceMoved.x));
}

TEST_CASE("searching becomes tracking only after the person stayed for two frames") {
    Rig r;
    r.step({ALICE});
    CHECK_FALSE(r.tracker.status().visible);
    CHECK(r.tracker.status().state == State::Searching);
    r.step({ALICE});
    CHECK(r.tracker.status().visible);
    CHECK(r.tracker.status().state == State::Tracking);
    CHECK(r.following(ALICE.x));
}

TEST_CASE("a person seen for one frame only is never followed") {
    Rig r;
    r.step({ALICE});
    r.step({});
    r.step({BOB});
    CHECK_FALSE(r.tracker.status().visible);
    CHECK(r.tracker.status().state == State::Searching);
}

TEST_CASE("target missing for one or two frames is still tracking but not visible") {
    Rig r;
    r.steps(15, {ALICE});
    REQUIRE(r.tracker.status().visible);
    for (int missed = 1; missed <= 2; missed++) {
        CAPTURE(missed);
        const auto& s = r.step({});
        CHECK(s.state == State::Tracking);
        CHECK_FALSE(s.visible);
    }
}

TEST_CASE("target missing for three frames is lost and comes back when seen again") {
    Rig r;
    r.steps(15, {ALICE});
    r.steps(2, {});
    CHECK(r.tracker.status().state == State::Tracking);
    r.step({});
    CHECK(r.tracker.status().state == State::Lost);
    CHECK_FALSE(r.tracker.status().visible);
    // the track is still kept, so the same person is simply followed again
    r.step({ALICE});
    CHECK(r.following(ALICE.x));
    CHECK(r.tracker.status().state == State::Tracking);
}

TEST_CASE("track of a missing target is deleted after more than eight missed frames") {
    Rig r;
    r.steps(15, {ALICE});
    REQUIRE(Probe::trackCount(r.tracker) == 1);
    r.steps(8, {});
    CHECK(Probe::trackCount(r.tracker) == 1);
    CHECK(r.tracker.status().state == State::Lost);
    r.step({});
    CHECK(Probe::trackCount(r.tracker) == 0);
    CHECK(Probe::targetId(r.tracker) == -1);
    CHECK(r.tracker.status().state == State::Lost);
}

TEST_CASE("dropTarget resets the status to searching") {
    Rig r;
    r.steps(15, {ALICE});
    REQUIRE(r.tracker.status().visible);
    r.tracker.dropTarget();
    const auto& s = r.tracker.status();
    CHECK(s.state == State::Searching);
    CHECK_FALSE(s.visible);
    CHECK(s.secondsSinceSeen == doctest::Approx(0));
    CHECK(s.predictedAngle == doctest::Approx(0));
    CHECK(Probe::targetId(r.tracker) == -1);
}

TEST_CASE("after dropTarget the same person is taken again by their clothes") {
    Rig r;
    r.steps(15, {ALICE});
    REQUIRE(Probe::clothingViews(r.tracker) > 0);
    r.tracker.dropTarget();
    r.step({ALICE});
    CHECK(r.following(ALICE.x));
    CHECK(r.tracker.status().state == State::Tracking);
}

TEST_CASE("after dropTarget a stranger in other clothes is not followed") {
    Rig r;
    r.steps(15, {ALICE});
    r.tracker.dropTarget();
    r.steps(10, {BOB});
    CHECK_FALSE(r.tracker.status().visible);
    CHECK(r.tracker.status().state != State::Tracking);
    // and the right person is still recognized next to them
    r.steps(3, {ALICE, BOB});
    CHECK(r.following(ALICE.x));
}

TEST_CASE("with reference photos nobody is followed without face evidence") {
    Rig r;
    Probe::setReferenceFace(r.tracker, referenceFeature());
    REQUIRE(r.tracker.hasReferencePhotos());
    REQUIRE(Probe::galleryFaces(r.tracker) == 1);
    r.steps(40, {ALICE, BOB});
    CHECK_FALSE(r.tracker.status().visible);
    CHECK(r.tracker.status().state == State::Searching);
    // evidence below the identification threshold is not enough either
    Probe::setFaceEvidence(r.tracker, ALICE.x, 0.5f);
    r.steps(5, {ALICE, BOB});
    CHECK_FALSE(r.tracker.status().visible);
    CHECK(r.tracker.status().state == State::Searching);
}

TEST_CASE("with reference photos the person with face evidence is followed on the next frame") {
    Rig r;
    Probe::setReferenceFace(r.tracker, referenceFeature());
    r.steps(5, {ALICE, BOB});
    REQUIRE_FALSE(r.tracker.status().visible);
    Probe::setFaceEvidence(r.tracker, BOB.x, 0.9f);
    Probe::setFaceEvidence(r.tracker, ALICE.x, 0.5f);
    r.step({ALICE, BOB});
    CHECK(r.following(BOB.x));
    CHECK(r.tracker.status().state == State::Tracking);
}

TEST_CASE("face evidence decays by one percent per frame without a face") {
    Rig r;
    Probe::setReferenceFace(r.tracker, referenceFeature());
    r.step({ALICE});
    Probe::setFaceEvidence(r.tracker, ALICE.x, 0.5f);
    r.steps(10, {ALICE});
    CHECK(Probe::faceEvidence(r.tracker, ALICE.x) == doctest::Approx(0.5f * std::pow(0.99f, 10)).epsilon(0.001));
    // negative evidence fades toward zero as well
    Probe::setFaceEvidence(r.tracker, ALICE.x, -0.4f);
    r.step({ALICE});
    CHECK(Probe::faceEvidence(r.tracker, ALICE.x) == doctest::Approx(-0.396f).epsilon(0.001));
}

TEST_CASE("a clearly better face on someone else takes the target over") {
    Rig r;
    r.steps(15, {ALICE});
    r.steps(3, {ALICE, BOB});
    REQUIRE(r.following(ALICE.x));
    Probe::setFaceEvidence(r.tracker, ALICE.x, 0.5f);
    // 0.95 beats 0.5 by more than 0.3 even after one frame of decay
    Probe::setFaceEvidence(r.tracker, BOB.x, 0.95f);
    r.step({ALICE, BOB});
    CHECK(r.following(BOB.x));
}

TEST_CASE("a face better by less than the margin does not take the target over") {
    Rig r;
    r.steps(15, {ALICE});
    r.steps(3, {ALICE, BOB});
    REQUIRE(r.following(ALICE.x));
    Probe::setFaceEvidence(r.tracker, ALICE.x, 0.5f);
    // 0.75 is identified (>= 0.6) but only 0.25 above
    Probe::setFaceEvidence(r.tracker, BOB.x, 0.75f);
    r.step({ALICE, BOB});
    CHECK(r.following(ALICE.x));
}

TEST_CASE("a recent face match of the target protects them from a takeover") {
    Rig r;
    r.steps(15, {ALICE});
    r.steps(3, {ALICE, BOB});
    REQUIRE(r.following(ALICE.x));
    Probe::setFaceEvidence(r.tracker, ALICE.x, 0.5f);
    Probe::setFaceEvidence(r.tracker, BOB.x, 0.95f);
    // matched one second ago: still counts as them
    Probe::setLastFaceMatch(r.tracker, ALICE.x, r.now - 1s);
    r.step({ALICE, BOB});
    CHECK(r.following(ALICE.x));
}

TEST_CASE("a face match older than two seconds no longer protects the target") {
    Rig r;
    r.steps(15, {ALICE});
    r.steps(3, {ALICE, BOB});
    REQUIRE(r.following(ALICE.x));
    Probe::setFaceEvidence(r.tracker, ALICE.x, 0.5f);
    Probe::setFaceEvidence(r.tracker, BOB.x, 0.95f);
    Probe::setLastFaceMatch(r.tracker, ALICE.x, r.now - 3s);
    r.step({ALICE, BOB});
    CHECK(r.following(BOB.x));
}

namespace {
// someone walks sideways (+20 px per frame = 200 px/s at 100 ms frames) for a while, then
// is gone; returns the last box that was seen
cv::Rect2f walkThenVanish(Rig& r, float startX, float stepPx, int frames) {
    Person walker = ALICE;
    walker.x = startX;
    for (int i = 0; i < frames; i++) {
        walker.x += stepPx;
        r.step({walker});
    }
    return walker.detection().box;
}
}

TEST_CASE("prediction of a person walking out to the left follows them and stops after two seconds") {
    Rig r;
    const cv::Rect2f last = walkThenVanish(r, 600, -20, 15);
    REQUIRE(r.tracker.status().visible);
    // while seen, the expectation is where they are
    CHECK(r.tracker.status().predictedAngle == doctest::Approx(CameraGeometry::angle(last, FRAME_WIDTH)).epsilon(0.001));
    CHECK(r.tracker.status().predictedDistance == doctest::Approx(CameraGeometry::distance(last, FRAME_WIDTH)).epsilon(0.001));
    CHECK(r.tracker.status().secondsSinceSeen == doctest::Approx(0).epsilon(0.001));

    const float angleAtLoss = r.tracker.status().predictedAngle;
    r.steps(10, {});
    const auto atOneSecond = r.tracker.status();
    CHECK(atOneSecond.lostSide == 1);
    CHECK(atOneSecond.secondsSinceSeen == doctest::Approx(1.0f).epsilon(0.01));
    // they kept walking left (positive angle) and away along their line
    CHECK(atOneSecond.predictedAngle > angleAtLoss + 0.05f);
    CHECK(atOneSecond.predictedDistance > CameraGeometry::distance(last, FRAME_WIDTH));

    r.steps(10, {});
    const auto atTwoSeconds = r.tracker.status();
    CHECK(atTwoSeconds.predictedAngle > atOneSecond.predictedAngle);
    CHECK(atTwoSeconds.secondsSinceSeen == doctest::Approx(2.0f).epsilon(0.01));

    // beyond the horizon only the time grows
    r.steps(10, {});
    const auto& later = r.tracker.status();
    CHECK(later.secondsSinceSeen == doctest::Approx(3.0f).epsilon(0.01));
    CHECK(later.predictedAngle == doctest::Approx(atTwoSeconds.predictedAngle).epsilon(0.0001));
    CHECK(later.predictedDistance == doctest::Approx(atTwoSeconds.predictedDistance).epsilon(0.0001));
    CHECK(later.lostSide == 1);
}

TEST_CASE("prediction of a person walking out to the right says they went right") {
    Rig r;
    walkThenVanish(r, 500, 20, 15);
    REQUIRE(r.tracker.status().visible);
    const float angleAtLoss = r.tracker.status().predictedAngle;
    CHECK(angleAtLoss < -0.1f);
    r.steps(10, {});
    CHECK(r.tracker.status().lostSide == -1);
    CHECK(r.tracker.status().predictedAngle < angleAtLoss - 0.05f);
}

TEST_CASE("prediction of a standing person stays where they were") {
    Rig r;
    r.steps(15, {ALICE});
    const float angle = r.tracker.status().predictedAngle;
    const float distance = r.tracker.status().predictedDistance;
    r.steps(15, {});
    CHECK(r.tracker.status().predictedAngle == doctest::Approx(angle).epsilon(0.001));
    CHECK(r.tracker.status().predictedDistance == doctest::Approx(distance).epsilon(0.001));
    CHECK(r.tracker.status().secondsSinceSeen == doctest::Approx(1.5f).epsilon(0.01));
}

TEST_CASE("predicted walking speed is capped at one and a half meters per second") {
    Rig r;
    // far away and fast: 60 px per 100 ms is about 2.2 m/s at 2.2 m
    Person walker {1000, RED, JEANS, 0.5f};
    cv::Rect2f previous = walker.detection().box;
    for (int i = 0; i < 12; i++) {
        walker.x -= 60;
        r.step({walker});
    }
    const cv::Rect2f last = walker.detection().box;
    previous = last;
    previous.x += 60;
    const cv::Point2f a = worldPosition(previous), b = worldPosition(last);
    REQUIRE(std::hypot(b.x - a.x, b.y - a.y) / 0.1f > 1.5f);
    REQUIRE(r.tracker.status().visible);

    // past the horizon the prediction has moved by the horizon times the capped speed
    r.steps(30, {});
    const auto& s = r.tracker.status();
    const cv::Point2f predicted(s.predictedDistance * std::cos(s.predictedAngle),
                                s.predictedDistance * std::sin(s.predictedAngle));
    CHECK(std::hypot(predicted.x - b.x, predicted.y - b.y) == doctest::Approx(1.5f * 2.0f).epsilon(0.02));
}

TEST_CASE("a new person where the target was heading is taken within two seconds of losing them") {
    Rig r;
    // photos are known, so nothing else would pick someone without a face
    Probe::setReferenceFace(r.tracker, referenceFeature());
    Person alice = ALICE;
    alice.x = 200;
    r.step({alice});
    Probe::setFaceEvidence(r.tracker, alice.x, 0.9f);
    r.steps(4, {alice});
    REQUIRE(r.following(alice.x));
    // their track is kept for a while, the others only count after it died
    r.steps(10, {});
    REQUIRE(r.tracker.status().state == State::Lost);
    Person newcomer = BOB;
    newcomer.x = 260;
    r.step({newcomer});
    CHECK(r.following(newcomer.x));
}

TEST_CASE("a new person is not taken for the lost target after more than two seconds") {
    Rig r;
    Probe::setReferenceFace(r.tracker, referenceFeature());
    Person alice = ALICE;
    alice.x = 200;
    r.step({alice});
    Probe::setFaceEvidence(r.tracker, alice.x, 0.9f);
    r.steps(4, {alice});
    REQUIRE(r.following(alice.x));
    r.steps(25, {});
    Person newcomer = BOB;
    newcomer.x = 260;
    r.steps(5, {newcomer});
    CHECK_FALSE(r.tracker.status().visible);
}

TEST_CASE("a new person far from where the target was heading is not taken") {
    Rig r;
    Probe::setReferenceFace(r.tracker, referenceFeature());
    Person alice = ALICE;
    alice.x = 100;
    r.step({alice});
    Probe::setFaceEvidence(r.tracker, alice.x, 0.9f);
    r.steps(4, {alice});
    REQUIRE(r.following(alice.x));
    r.steps(10, {});
    // alice was on the left, this one is on the far right (well over 0.4 rad away)
    Person newcomer = BOB;
    newcomer.x = 1000;
    r.steps(3, {newcomer});
    CHECK_FALSE(r.tracker.status().visible);
}

TEST_CASE("a new person whose clothes contradict the target is not taken where they were heading") {
    Rig r;
    Person alice = ALICE;
    alice.x = 200;
    // long enough for the clothes to be learned
    r.steps(15, {alice});
    REQUIRE(Probe::clothingViews(r.tracker) > 0);
    r.steps(10, {});
    Person stranger = BOB;
    stranger.x = 200;
    r.steps(3, {stranger});
    CHECK_FALSE(r.tracker.status().visible);
}

TEST_CASE("at most three faces are checked per frame") {
    Rig r;
    std::vector<Person> crowd;
    const cv::Scalar shirts[] = {RED, GREEN, BLUE, YELLOW, PURPLE};
    for (int i = 0; i < 5; i++) crowd.push_back({20.0f + 250 * i, shirts[i], BLACK, 1.0f, true});
    r.steps(3, crowd);
    CHECK(Probe::trackCount(r.tracker) == 5);
    // checked even though no face is found without the detector
    CHECK(Probe::facesChecked(r.tracker) == 3);
}

TEST_CASE("all faces are checked when there are fewer people than the limit") {
    Rig r;
    r.steps(3, {{100, RED, BLACK, 1.0f, true}, {600, GREEN, BLACK, 1.0f, true}});
    CHECK(Probe::facesChecked(r.tracker) == 2);
}

TEST_CASE("people seen from behind get no face check") {
    Rig r;
    r.steps(3, {{100, RED, BLACK, 1.0f, false}, {600, GREEN, BLACK, 1.0f, true}, {900, BLUE, BLACK, 1.0f, false}});
    CHECK(Probe::facesChecked(r.tracker) == 1);
    CHECK(Probe::faceChecked(r.tracker, 600));
    CHECK_FALSE(Probe::faceChecked(r.tracker, 100));
}

TEST_CASE("the target's face is checked first even when they are the smallest") {
    Rig r;
    // the first person is followed, and they are the farthest away
    std::vector<Person> crowd = {{20, RED, BLACK, 0.6f, true}};
    const cv::Scalar shirts[] = {GREEN, BLUE, YELLOW, PURPLE};
    for (int i = 0; i < 4; i++) crowd.push_back({250.0f + 250 * i, shirts[i], BLACK, 1.0f, true});
    r.steps(4, crowd);
    REQUIRE(r.following(20));
    CHECK(Probe::facesChecked(r.tracker) == 3);
    CHECK(Probe::faceChecked(r.tracker, 20));
}

TEST_CASE("a pause longer than half a second forgets all tracks") {
    Rig r;
    r.steps(15, {ALICE, BOB});
    REQUIRE(Probe::trackCount(r.tracker) == 2);
    const int before = Probe::targetId(r.tracker);
    Person carol {100, BLUE, BLACK};
    r.step({carol}, 600ms);
    // only the person in the new frame is tracked, the old ones are gone
    CHECK(Probe::trackCount(r.tracker) == 1);
    CHECK(Probe::targetId(r.tracker) != before);
    CHECK_FALSE(r.tracker.status().visible);
}

TEST_CASE("a pause of less than half a second keeps the tracks") {
    Rig r;
    r.steps(15, {ALICE, BOB});
    REQUIRE(r.following(ALICE.x));
    r.step({ALICE, BOB}, 400ms);
    CHECK(Probe::trackCount(r.tracker) == 2);
    CHECK(r.following(ALICE.x));
}

TEST_CASE("drawDebug leaves an empty image alone") {
    Rig r;
    r.steps(15, {ALICE});
    cv::Mat empty;
    CHECK_NOTHROW(r.tracker.drawDebug(empty));
    CHECK(empty.empty());
}

TEST_CASE("drawDebug draws the target and the status into the frame") {
    Rig r;
    r.steps(15, {ALICE});
    REQUIRE(r.tracker.status().visible);
    const cv::Mat frame = drawFrame({ALICE});
    cv::Mat image = frame.clone();
    CHECK_NOTHROW(r.tracker.drawDebug(image));
    CHECK(image.size() == frame.size());
    cv::Mat difference;
    cv::absdiff(image, frame, difference);
    CHECK(cv::countNonZero(difference.reshape(1)) > 0);
}
