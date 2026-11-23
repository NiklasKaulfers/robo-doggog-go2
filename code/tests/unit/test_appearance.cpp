// Appearance: clothing histograms, comparing them and the learned model of the target's clothes
#include <cmath>
#include <numeric>
#include <vector>
#include "doctest/doctest.h"
#include "opencv2/imgproc.hpp"
#include "DrawnPerson.h"
#include "ml/Appearance.h"

using robo::sdk::ml::Appearance;
using robo::sdk::ml::PersonDetection;
using namespace testutil;

namespace {

constexpr size_t BINS = 28;

float sum(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f);
}

// a frame of one flat colour with a drawn person's keypoints on it
Appearance::Signature describeFlat(const cv::Scalar& colour) {
    const cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, colour);
    Appearance::Signature signature;
    REQUIRE(Appearance::describe(frame, Person{500, RED, JEANS}.detection(), signature));
    REQUIRE(signature.torso.size() == BINS);
    return signature;
}

// a person with a box and keypoints given as {index, x, y}, all confident
struct Pose {
    PersonDetection detection;

    explicit Pose(cv::Rect2f box) { detection.box = box; }

    Pose& with(int keypoint, float x, float y, float confidence = 0.9f) {
        detection.keypoints[keypoint] = {x, y};
        detection.confidence[keypoint] = confidence;
        return *this;
    }
};

const cv::Mat BLANK(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cv::Scalar(100, 100, 100));

cv::Rect torsoOf(const Pose& pose, const cv::Mat& frame = BLANK) {
    cv::Rect torso, legs;
    Appearance::regions(frame, pose.detection, torso, legs);
    return torso;
}

cv::Rect legsOf(const Pose& pose, const cv::Mat& frame = BLANK) {
    cv::Rect torso, legs;
    Appearance::regions(frame, pose.detection, torso, legs);
    return legs;
}

// shoulders, hips, knees and ankles of a standing person at known places
Pose standing(float confidenceOfFeet = 0.9f) {
    using P = PersonDetection;
    Pose pose(cv::Rect2f(350, 50, 300, 600));
    pose.with(P::LEFT_SHOULDER, 400, 100).with(P::RIGHT_SHOULDER, 600, 100)
        .with(P::LEFT_HIP, 420, 300).with(P::RIGHT_HIP, 580, 300)
        .with(P::LEFT_KNEE, 425, 450).with(P::RIGHT_KNEE, 575, 450)
        .with(P::LEFT_ANKLE, 430, 600, confidenceOfFeet).with(P::RIGHT_ANKLE, 570, 600, confidenceOfFeet);
    return pose;
}

// a signature with all the mass of the torso in one bin
Appearance::Signature oneHot(size_t bin) {
    Appearance::Signature signature;
    signature.torso.assign(BINS, 0.0f);
    signature.torso[bin] = 1;
    return signature;
}

// a signature with the mass split between two bins of the torso
Appearance::Signature twoBins(size_t first, float firstShare, size_t second) {
    Appearance::Signature signature;
    signature.torso.assign(BINS, 0.0f);
    signature.torso[first] = firstShare;
    signature.torso[second] = 1 - firstShare;
    return signature;
}
}

TEST_CASE("a drawn person's clothes look like themselves") {
    const Person alice {500, RED, JEANS};
    Appearance::Signature signature;
    REQUIRE(Appearance::describe(drawFrame({alice}), alice.detection(), signature));
    CHECK(Appearance::compare(signature, signature) == doctest::Approx(1.0f));
}

TEST_CASE("an empty frame describes nothing") {
    Appearance::Signature signature;
    signature.torso = {1.0f};
    CHECK_FALSE(Appearance::describe(cv::Mat(), Person{500, RED, JEANS}.detection(), signature));
    CHECK(signature.empty());
}

TEST_CASE("a fully visible person gets normalized torso and legs histograms") {
    const Person alice {500, RED, JEANS};
    Appearance::Signature signature;
    REQUIRE(Appearance::describe(drawFrame({alice}), alice.detection(), signature));
    for (const auto* hist : {&signature.torso, &signature.legs}) {
        REQUIRE(hist->size() == BINS);
        for (float bin : *hist) CHECK(bin >= 0);
        CHECK(sum(*hist) == doctest::Approx(1.0f).epsilon(1e-4));
    }
}

TEST_CASE("a black region lands in the dark bin") {
    const auto signature = describeFlat(cv::Scalar(10, 10, 10));
    CHECK(signature.torso[0] == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("gray regions land in the gray bins by brightness") {
    // value 80 is the darkest gray bin, value 220 the brightest
    const auto dark = describeFlat(cv::Scalar(80, 80, 80));
    const auto light = describeFlat(cv::Scalar(220, 220, 220));
    CHECK(dark.torso[1] == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(light.torso[3] == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(Appearance::compare(dark, light) == doctest::Approx(0.0f).epsilon(1e-4));
}

TEST_CASE("a saturated colour only fills hue bins") {
    const auto signature = describeFlat(cv::Scalar(0, 255, 0));
    float hueMass = 0;
    for (size_t i = 4; i < BINS; i++) hueMass += signature.torso[i];
    CHECK(hueMass == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("the saturation of a colour picks the sub bin of its hue") {
    // hue 30 sits between hue bins 1 and 2, saturation 128 is the low and 255 the high sub bin
    const auto pale = describeFlat(cv::Scalar(100, 200, 200));
    const auto vivid = describeFlat(cv::Scalar(0, 255, 255));
    CHECK(pale.torso[6] + pale.torso[8] == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(vivid.torso[7] + vivid.torso[9] == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("a hue between two bin centres is split half and half") {
    // hue 30 is exactly the border of hue bins 1 and 2
    const auto signature = describeFlat(cv::Scalar(0, 255, 255));
    CHECK(signature.torso[7] == doctest::Approx(0.5f).epsilon(1e-3));
    CHECK(signature.torso[9] == doctest::Approx(0.5f).epsilon(1e-3));
}

TEST_CASE("red wraps around between the last and the first hue bin") {
    // hue 0 is the border of hue bins 11 and 0, hue is circular
    const auto signature = describeFlat(cv::Scalar(0, 0, 255));
    CHECK(signature.torso[4 + 11 * 2 + 1] == doctest::Approx(0.5f).epsilon(1e-3));
    CHECK(signature.torso[4 + 0 * 2 + 1] == doctest::Approx(0.5f).epsilon(1e-3));
}

TEST_CASE("identical clothes compare as 1") {
    const Person alice {500, RED, JEANS};
    Appearance::Signature a, b;
    REQUIRE(Appearance::describe(drawFrame({alice}), alice.detection(), a));
    REQUIRE(Appearance::describe(drawFrame({alice}), alice.detection(), b));
    CHECK(Appearance::compare(a, b) == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("a red and a green shirt are clearly less alike than the same shirt") {
    const Person red {500, RED, JEANS}, green {500, GREEN, JEANS};
    Appearance::Signature a, b;
    REQUIRE(Appearance::describe(drawFrame({red}), red.detection(), a));
    REQUIRE(Appearance::describe(drawFrame({green}), green.detection(), b));
    CAPTURE(Appearance::compare(a, b));
    CHECK(Appearance::compare(a, b) < 0.6f);
}

TEST_CASE("signatures with no region in common compare as minus one") {
    Appearance::Signature torsoOnly = oneHot(0), legsOnly;
    legsOnly.legs.assign(BINS, 1.0f / BINS);
    CHECK(Appearance::compare(torsoOnly, legsOnly) == -1.0f);
    CHECK(Appearance::compare(legsOnly, torsoOnly) == -1.0f);
}

TEST_CASE("two empty signatures compare as minus one") {
    CHECK(Appearance::compare({}, {}) == -1.0f);
}

TEST_CASE("only the regions present in both signatures count") {
    Appearance::Signature both = twoBins(0, 0.5f, 1), torsoOnly = twoBins(0, 0.25f, 1);
    both.legs.assign(BINS, 1.0f / BINS);
    const float torsoOverlap = std::sqrt(0.5f * 0.25f) + std::sqrt(0.5f * 0.75f);
    CHECK(Appearance::compare(both, torsoOnly) == doctest::Approx(torsoOverlap));
    // the legs of the other person are not compared against nothing
    CHECK(Appearance::compare(torsoOnly, both) == doctest::Approx(torsoOverlap));
}

TEST_CASE("with both regions the torso counts 60 percent and the legs 40") {
    Appearance::Signature a = twoBins(0, 0.5f, 1), b = twoBins(0, 0.2f, 1);
    a.legs.assign(BINS, 0.0f);
    b.legs.assign(BINS, 0.0f);
    a.legs[5] = 0.64f; a.legs[6] = 0.36f;
    b.legs[5] = 0.16f; b.legs[6] = 0.84f;
    const float torso = std::sqrt(0.5f * 0.2f) + std::sqrt(0.5f * 0.8f);
    const float legs = std::sqrt(0.64f * 0.16f) + std::sqrt(0.36f * 0.84f);
    CHECK(Appearance::compare(a, b) == doctest::Approx(0.6f * torso + 0.4f * legs));
}

TEST_CASE("the overlap never exceeds 1") {
    Appearance::Signature a;
    a.torso.assign(BINS, 0.0f);
    a.torso[0] = 0.6f;
    a.torso[1] = 0.6f;
    CHECK(Appearance::compare(a, a) == 1.0f);
}

TEST_CASE("the torso spans shoulders and hips narrowed on both sides") {
    using P = PersonDetection;
    Pose pose(cv::Rect2f(350, 50, 300, 600));
    pose.with(P::LEFT_SHOULDER, 400, 200).with(P::RIGHT_SHOULDER, 600, 200)
        .with(P::LEFT_HIP, 420, 500).with(P::RIGHT_HIP, 580, 500);
    // width 200 narrowed by 24 each side, top 10 percent of the 300 px below the shoulders
    CHECK(torsoOf(pose) == cv::Rect(424, 230, 152, 270));
}

TEST_CASE("without hips the torso is 1.2 shoulder widths long") {
    using P = PersonDetection;
    Pose pose(cv::Rect2f(350, 100, 300, 500));
    pose.with(P::LEFT_SHOULDER, 400, 200).with(P::RIGHT_SHOULDER, 600, 200);
    // bottom at 200 + 240, top 10 percent of that below the shoulders
    CHECK(torsoOf(pose) == cv::Rect(424, 224, 152, 216));
}

TEST_CASE("without hips the torso stops at the bottom of the box") {
    using P = PersonDetection;
    Pose pose(cv::Rect2f(350, 100, 300, 300));
    pose.with(P::LEFT_SHOULDER, 400, 200).with(P::RIGHT_SHOULDER, 600, 200);
    CHECK(torsoOf(pose) == cv::Rect(424, 220, 152, 180));
}

TEST_CASE("the legs go from the hips to the ankles widened by 30 percent") {
    const auto pose = standing();
    // hips 160 px apart, widened by 48 each side
    CHECK(legsOf(pose) == cv::Rect(372, 300, 256, 300));
}

TEST_CASE("the legs end at the knees when the ankles are missing") {
    CHECK(legsOf(standing(0.0f)) == cv::Rect(372, 300, 256, 150));
}

TEST_CASE("keypoints below the confidence limit are ignored") {
    // ankles at 0.1 count as missing, so the legs end at the knees
    CHECK(legsOf(standing(0.1f)) == cv::Rect(372, 300, 256, 150));

    // unsure hips: the torso is placed from the shoulders alone and no legs are found
    auto pose = standing();
    pose.detection.confidence[PersonDetection::LEFT_HIP] = 0.1f;
    CHECK(torsoOf(pose) == torsoOf([] {
        Pose p = standing();
        p.detection.confidence[PersonDetection::LEFT_HIP] = 0;
        p.detection.confidence[PersonDetection::RIGHT_HIP] = 0;
        return p;
    }()));
    CHECK(legsOf(pose).area() == 0);
}

TEST_CASE("a tall box without keypoints gets the fallback regions") {
    const Pose pose(cv::Rect2f(400, 100, 200, 400));
    // half of the width in the middle, torso at 20 to 50 percent, legs at 55 to 90
    CHECK(torsoOf(pose) == cv::Rect(450, 180, 100, 120));
    CHECK(legsOf(pose) == cv::Rect(450, 320, 100, 140));
}

TEST_CASE("a wide box without keypoints gets no regions") {
    const Pose pose(cv::Rect2f(400, 100, 400, 300));
    CHECK(torsoOf(pose).area() == 0);
    CHECK(legsOf(pose).area() == 0);
    Appearance::Signature signature;
    CHECK_FALSE(Appearance::describe(BLANK, pose.detection, signature));
    CHECK(signature.empty());
}

TEST_CASE("regions are clipped to the frame") {
    // half of the person is left of the frame
    CHECK(torsoOf(Pose(cv::Rect2f(-100, 100, 200, 400))) == cv::Rect(0, 180, 50, 120));
    // the torso is cut off by the bottom and the legs hang below the frame
    const Pose low(cv::Rect2f(400, 600, 200, 400));
    CHECK(torsoOf(low) == cv::Rect(450, 680, 100, 40));
    CHECK(legsOf(low).area() == 0);
    // a frame smaller than the person
    const cv::Mat small(400, 1280, CV_8UC3, cv::Scalar(100, 100, 100));
    CHECK(legsOf(Pose(cv::Rect2f(400, 100, 200, 400)), small) == cv::Rect(450, 320, 100, 80));
}

TEST_CASE("a region under 150 pixels gives an empty histogram") {
    using P = PersonDetection;
    Pose pose(cv::Rect2f(50, 50, 400, 400));
    pose.with(P::LEFT_SHOULDER, 100, 100).with(P::RIGHT_SHOULDER, 300, 100)
        .with(P::LEFT_HIP, 200, 300).with(P::RIGHT_HIP, 204, 300)
        .with(P::LEFT_ANKLE, 200, 302).with(P::RIGHT_ANKLE, 204, 302);
    CAPTURE(legsOf(pose).area());
    REQUIRE(legsOf(pose).area() < 150);
    Appearance::Signature signature;
    REQUIRE(Appearance::describe(BLANK, pose.detection, signature));
    CHECK(signature.torso.size() == BINS);
    CHECK(signature.legs.empty());
}

TEST_CASE("a region over 40000 pixels still gives a normalized histogram") {
    const Pose pose(cv::Rect2f(0, 0, 500, 700));
    REQUIRE(torsoOf(pose).area() > 40000);
    Appearance::Signature signature;
    REQUIRE(Appearance::describe(BLANK, pose.detection, signature));
    REQUIRE(signature.torso.size() == BINS);
    CHECK(sum(signature.torso) == doctest::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("a new appearance model is empty") {
    Appearance model;
    CHECK(model.empty());
    CHECK(model.size() == 0);
    CHECK(model.anchors() == 0);
    CHECK(model.similarity(oneHot(0)) == -1.0f);
}

TEST_CASE("learning an empty signature does nothing") {
    Appearance model;
    model.learn({}, true);
    CHECK(model.empty());
}

TEST_CASE("the similarity is the best match over all learned views") {
    Appearance model;
    model.learn(oneHot(0), false);
    model.learn(oneHot(1), false);
    REQUIRE(model.size() == 2);
    CHECK(model.similarity(twoBins(0, 0.8f, 1)) == doctest::Approx(std::sqrt(0.8f)));
    CHECK(model.similarity(oneHot(2)) == doctest::Approx(0.0f));
}

TEST_CASE("a near duplicate view is not added") {
    Appearance model;
    model.learn(oneHot(0), false);
    model.learn(twoBins(0, 0.99f, 1), false);
    CHECK(model.size() == 1);
}

TEST_CASE("a view just below the duplicate limit is added") {
    Appearance model;
    model.learn(oneHot(0), false);
    // overlap sqrt(0.9) = 0.949
    model.learn(twoBins(0, 0.9f, 1), false);
    CHECK(model.size() == 2);
}

TEST_CASE("a near duplicate learned as anchor promotes the existing view") {
    Appearance model;
    model.learn(oneHot(0), false);
    REQUIRE(model.anchors() == 0);
    model.learn(twoBins(0, 0.99f, 1), true);
    CHECK(model.size() == 1);
    CHECK(model.anchors() == 1);
}

TEST_CASE("only the 12 newest non anchor views are kept") {
    Appearance model;
    for (size_t i = 0; i < 13; i++) model.learn(oneHot(i), false);
    CHECK(model.size() == 12);
    CHECK(model.similarity(oneHot(0)) == doctest::Approx(0.0f));
    for (size_t i = 1; i < 13; i++) CHECK(model.similarity(oneHot(i)) == doctest::Approx(1.0f));
}

TEST_CASE("anchors are not pushed out by non anchor views") {
    Appearance model;
    model.learn(oneHot(0), true);
    for (size_t i = 1; i <= 20; i++) model.learn(oneHot(i), false);
    CHECK(model.anchors() == 1);
    CHECK(model.size() == 12);
    CHECK(model.similarity(oneHot(0)) == doctest::Approx(1.0f));
}

TEST_CASE("only the 6 newest anchors are kept") {
    Appearance model;
    for (size_t i = 0; i < 7; i++) model.learn(oneHot(i), true);
    CHECK(model.anchors() == 6);
    CHECK(model.size() == 6);
    CHECK(model.similarity(oneHot(0)) == doctest::Approx(0.0f));
    CHECK(model.similarity(oneHot(6)) == doctest::Approx(1.0f));
}

TEST_CASE("clearing the model forgets everything") {
    Appearance model;
    model.learn(oneHot(0), true);
    model.learn(oneHot(1), false);
    model.clear();
    CHECK(model.empty());
    CHECK(model.anchors() == 0);
    CHECK(model.similarity(oneHot(0)) == -1.0f);
}
