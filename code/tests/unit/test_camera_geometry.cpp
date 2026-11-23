// CameraGeometry (pixels -> angle and distance) and PersonDetection
#include "doctest/doctest.h"
#include "tracking/CameraGeometry.h"
#include "ml/PersonDetection.h"

#include <cmath>

using robo::sdk::tracking::CameraGeometry;
using robo::sdk::ml::PersonDetection;

namespace {
constexpr float kPi = 3.14159265f;
constexpr int kWidth = 1280;

// a zero-width box whose centre sits at the given fraction of the frame width
cv::Rect2f boxAtFraction(float fraction) {
    return cv::Rect2f(fraction * kWidth, 100, 0, 300);
}
}

TEST_CASE("angle is 0 for a box in the middle of the frame") {
    CHECK(CameraGeometry::angle(cv::Rect2f(590, 100, 100, 300), kWidth) == doctest::Approx(0.0f));
}

TEST_CASE("angle is plus half the field of view for a box at the left edge") {
    CHECK(CameraGeometry::angle(boxAtFraction(0.0f), kWidth) == doctest::Approx(CameraGeometry::FOV / 2));
}

TEST_CASE("angle is minus half the field of view for a box at the right edge") {
    CHECK(CameraGeometry::angle(boxAtFraction(1.0f), kWidth) == doctest::Approx(-CameraGeometry::FOV / 2));
}

TEST_CASE("angle is positive when the person is left of centre") {
    // positive means left, so a box in the left half must give a positive angle
    CHECK(CameraGeometry::angle(boxAtFraction(0.25f), kWidth) > 0.0f);
    CHECK(CameraGeometry::angle(boxAtFraction(0.75f), kWidth) < 0.0f);
}

TEST_CASE("angle is linear in the horizontal box centre") {
    // a quarter of the frame to the left should be exactly a quarter of the field of view
    CHECK(CameraGeometry::angle(boxAtFraction(0.25f), kWidth) == doctest::Approx(CameraGeometry::FOV / 4));
    CHECK(CameraGeometry::angle(boxAtFraction(0.75f), kWidth) == doctest::Approx(-CameraGeometry::FOV / 4));
}

TEST_CASE("angle ignores box width height and vertical position") {
    // only the horizontal centre matters, so these boxes share the same centre x
    const float expected = CameraGeometry::angle(cv::Rect2f(600, 0, 80, 80), kWidth);
    CHECK(CameraGeometry::angle(cv::Rect2f(560, 300, 160, 40), kWidth) == doctest::Approx(expected));
    CHECK(CameraGeometry::angle(cv::Rect2f(440, 10, 400, 700), kWidth) == doctest::Approx(expected));
}

TEST_CASE("distance equals the calibration distance at the calibration width") {
    const float width = CameraGeometry::BOX_WIDTH_AT_CALIBRATION * kWidth;
    CHECK(CameraGeometry::distance(cv::Rect2f(100, 0, width, 300), kWidth)
          == doctest::Approx(CameraGeometry::CALIBRATION_DISTANCE));
}

TEST_CASE("distance doubles when the box width halves") {
    // apparent width is inversely proportional to distance
    const float full = CameraGeometry::BOX_WIDTH_AT_CALIBRATION * kWidth;
    const float d1 = CameraGeometry::distance(cv::Rect2f(100, 0, full, 300), kWidth);
    const float d2 = CameraGeometry::distance(cv::Rect2f(100, 0, full / 2, 300), kWidth);
    CHECK(d2 == doctest::Approx(2 * d1));
}

TEST_CASE("distance clamps a zero width box to a finite large value") {
    // the width is floored at 1e-3 of the frame so the result never becomes inf
    const float expected = CameraGeometry::CALIBRATION_DISTANCE * CameraGeometry::BOX_WIDTH_AT_CALIBRATION / 1e-3f;
    const float d = CameraGeometry::distance(cv::Rect2f(100, 0, 0, 300), kWidth);
    CHECK(std::isfinite(d));
    CHECK(d == doctest::Approx(expected));
}

TEST_CASE("distance clamps a negative width box to the same value as zero width") {
    const float expected = CameraGeometry::CALIBRATION_DISTANCE * CameraGeometry::BOX_WIDTH_AT_CALIBRATION / 1e-3f;
    const float d = CameraGeometry::distance(cv::Rect2f(100, 0, -10, 300), kWidth);
    CHECK(std::isfinite(d));
    CHECK(d == doctest::Approx(expected));
}

TEST_CASE("pixel shift for a full field of view turn is the frame width") {
    CHECK(CameraGeometry::pixelShift(CameraGeometry::FOV, kWidth) == doctest::Approx(static_cast<float>(kWidth)));
}

TEST_CASE("pixel shift is zero for zero yaw change") {
    CHECK(CameraGeometry::pixelShift(0.0f, kWidth) == doctest::Approx(0.0f));
}

TEST_CASE("pixel shift takes the sign of the yaw change") {
    CHECK(CameraGeometry::pixelShift(0.1f, kWidth) > 0.0f);
    CHECK(CameraGeometry::pixelShift(-0.1f, kWidth) < 0.0f);
}

TEST_CASE("pixel shift is linear in the yaw change") {
    CHECK(CameraGeometry::pixelShift(0.4f, kWidth) == doctest::Approx(2 * CameraGeometry::pixelShift(0.2f, kWidth)));
}

TEST_CASE("shifting a box by the pixel shift of a turn changes its angle by minus that turn") {
    // a robot turning left by d sees a standing person d further to the right
    const float turns[] = {-0.3f, -0.05f, 0.05f, 0.2f, 0.7f};
    const float xs[] = {0.1f, 0.4f, 0.8f};
    for (float turn : turns) {
        for (float x : xs) {
            const cv::Rect2f before(x * kWidth, 100, 60, 200);
            cv::Rect2f after = before;
            after.x += CameraGeometry::pixelShift(turn, kWidth);
            const float delta = CameraGeometry::angle(after, kWidth) - CameraGeometry::angle(before, kWidth);
            CAPTURE(turn);
            CAPTURE(x);
            CHECK(delta == doctest::Approx(-turn).epsilon(1e-4));
        }
    }
}

TEST_CASE("wrap angle leaves zero unchanged") {
    CHECK(CameraGeometry::wrapAngle(0.0f) == doctest::Approx(0.0f));
}

TEST_CASE("wrap angle leaves a right angle unchanged") {
    CHECK(CameraGeometry::wrapAngle(kPi / 2) == doctest::Approx(kPi / 2));
}

TEST_CASE("wrap angle removes a full turn") {
    CHECK(CameraGeometry::wrapAngle(2 * kPi + 0.1f) == doctest::Approx(0.1f).epsilon(1e-4));
}

TEST_CASE("wrap angle maps three quarter turns back to a negative right angle equivalent") {
    CHECK(CameraGeometry::wrapAngle(-3 * kPi / 2) == doctest::Approx(kPi / 2).epsilon(1e-4));
}

TEST_CASE("wrap angle of three pi has magnitude pi") {
    // pi and -pi are the same heading, so only the magnitude is pinned here
    CHECK(std::fabs(CameraGeometry::wrapAngle(3 * kPi)) == doctest::Approx(kPi).epsilon(1e-4));
}

TEST_CASE("wrap angle keeps every sampled input inside minus pi to pi") {
    for (int i = -200; i <= 200; ++i) {
        const float a = static_cast<float>(i) / 10.0f;
        const float w = CameraGeometry::wrapAngle(a);
        CAPTURE(a);
        CHECK(w >= -kPi - 1e-6f);
        CHECK(w <= kPi + 1e-6f);
    }
}

TEST_CASE("person detection has seventeen keypoints") {
    CHECK(PersonDetection::KEYPOINT_COUNT == 17);
}

TEST_CASE("person detection keypoint enum follows the coco order") {
    // the order must match the Movenet model output
    CHECK(PersonDetection::NOSE == 0);
    CHECK(PersonDetection::LEFT_SHOULDER == 5);
    CHECK(PersonDetection::RIGHT_HIP == 12);
    CHECK(PersonDetection::RIGHT_ANKLE == 16);
}

TEST_CASE("person detection has a keypoint when confidence equals the minimum") {
    PersonDetection p;
    p.confidence[PersonDetection::LEFT_SHOULDER] = 0.5f;
    CHECK(p.has(PersonDetection::LEFT_SHOULDER, 0.5f));
    CHECK(p.has(PersonDetection::LEFT_SHOULDER, 0.4f));
    CHECK_FALSE(p.has(PersonDetection::LEFT_SHOULDER, 0.6f));
}

TEST_CASE("default person detection has no keypoints above a positive threshold") {
    // a default detection has all confidences at 0, so only a zero threshold passes
    PersonDetection p;
    CHECK_FALSE(p.has(PersonDetection::NOSE, 0.1f));
    CHECK(p.has(PersonDetection::NOSE, 0.0f));
}

TEST_CASE("person detection center is the centre of its box") {
    PersonDetection p;
    p.box = cv::Rect2f(10, 20, 30, 40);
    CHECK(p.center().x == doctest::Approx(25.0f));
    CHECK(p.center().y == doctest::Approx(40.0f));
}
