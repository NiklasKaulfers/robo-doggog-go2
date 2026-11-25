// Self-test of Appearance (the clothes model) with drawn people: the body regions chosen
// from the keypoints and their fallbacks, the color signatures and how they compare, large
// regions, and the set of learned views (duplicates, anchors, eviction). Runs without the
// robot and without the models:
//
//   cd code && ./build/appearancetest
//
// Exit code 0 when all checks pass.

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "../src/ml/Appearance.h"
#include "../src/ml/PersonDetection.h"

using namespace robo::sdk::ml;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok    " : "  FAIL  ") << what << std::endl;
    if (!condition) failures++;
}

static void setPoint(PersonDetection& p, int k, float x, float y) {
    p.keypoints[k] = cv::Point2f(x, y);
    p.confidence[k] = 0.9f;
}

// a person 200 x 600 px with its left edge at x and top at y; which keypoints are set varies
static PersonDetection person(float x, float y, bool shoulders, bool hips, bool knees, bool ankles) {
    PersonDetection p;
    p.box = cv::Rect2f(x, y, 200, 600);
    p.score = 0.8f;
    if (shoulders) {
        setPoint(p, PersonDetection::LEFT_SHOULDER, x + 40, y + 120);
        setPoint(p, PersonDetection::RIGHT_SHOULDER, x + 160, y + 120);
    }
    if (hips) {
        setPoint(p, PersonDetection::LEFT_HIP, x + 60, y + 320);
        setPoint(p, PersonDetection::RIGHT_HIP, x + 140, y + 320);
    }
    if (knees) {
        setPoint(p, PersonDetection::LEFT_KNEE, x + 65, y + 460);
        setPoint(p, PersonDetection::RIGHT_KNEE, x + 135, y + 460);
    }
    if (ankles) {
        setPoint(p, PersonDetection::LEFT_ANKLE, x + 65, y + 590);
        setPoint(p, PersonDetection::RIGHT_ANKLE, x + 135, y + 590);
    }
    return p;
}

// a frame with the shirt and trousers of the person in these colors
static cv::Mat dress(const cv::Size& size, const PersonDetection& p, cv::Scalar shirt, cv::Scalar trousers) {
    cv::Mat frame(size, CV_8UC3, cv::Scalar(120, 120, 120));
    const auto& b = p.box;
    cv::rectangle(frame, cv::Rect(b.x + 30, b.y + 100, 140, 230), shirt, cv::FILLED);
    cv::rectangle(frame, cv::Rect(b.x + 40, b.y + 330, 120, 270), trousers, cv::FILLED);
    return frame;
}

static float sum(const std::vector<float>& v) {
    float s = 0;
    for (float f : v) s += f;
    return s;
}

// a signature with all the weight on one bin, so different bins don't overlap at all
static Appearance::Signature view(size_t bin) {
    Appearance::Signature s;
    s.torso.assign(29, 0);
    s.torso[bin] = 1;
    return s;
}

int main() {
    const cv::Scalar red(40, 40, 220), blue(160, 60, 30), green(60, 180, 60), black(30, 30, 30);
    const cv::Size size(1280, 1000);

    std::cout << "1. regions from the full set of keypoints" << std::endl;
    {
        const auto p = person(500, 100, true, true, true, true);
        const cv::Mat frame = dress(size, p, red, blue);
        cv::Rect torso, legs;
        Appearance::regions(frame, p, torso, legs);
        // between the shoulders and hips, narrowed by 12% on each side, starting 10% below the shoulders
        check(torso.y > 220 && torso.y < 260 && torso.br().y == 420, "torso lies between the shoulders and the hips");
        check(torso.x > 540 && torso.br().x < 660, "torso is narrower than the shoulders");
        check(legs.y == 420 && legs.br().y == 690, "legs run from the hips to the ankles");
        check(legs.x < 560 && legs.br().x > 640, "legs are widened beyond the keypoints");

        Appearance::Signature s;
        check(Appearance::describe(frame, p, s), "describe succeeds");
        check(std::abs(sum(s.torso) - 1) < 1e-3 && std::abs(sum(s.legs) - 1) < 1e-3, "histograms are normalized");
        check(!s.empty(), "signature is not empty");
        check(!Appearance::describe(cv::Mat(), p, s) && s.empty(), "an empty frame describes nothing");
    }

    std::cout << "2. legs end at the knees when the feet are cut off" << std::endl;
    {
        const auto p = person(500, 100, true, true, true, false);
        cv::Rect torso, legs;
        Appearance::regions(dress(size, p, red, blue), p, torso, legs);
        check(legs.y == 420 && legs.br().y == 560, "legs end at the knees");
    }

    std::cout << "3. shoulders but no hips: the torso length comes from the shoulder width" << std::endl;
    {
        const auto p = person(500, 100, true, false, false, false);
        cv::Rect torso, legs;
        Appearance::regions(dress(size, p, red, blue), p, torso, legs);
        // shoulders at y 220, 120 px apart: the torso reaches 1.2 * 120 = 144 px lower
        check(torso.br().y == 364, "torso reaches 1.2 shoulder widths below the shoulders (bottom " + std::to_string(torso.br().y) + ")");
        check(torso.y > 220 && torso.y < 240, "torso starts a bit below the collar (top " + std::to_string(torso.y) + ")");
        check(torso.x == 554 && torso.br().x == 646, "torso is narrowed from the shoulders (" + std::to_string(torso.x) + ".." + std::to_string(torso.br().x) + ")");
        check(legs.area() == 0, "no legs without hips (even for a whole-body box)");

        // the box ends just below the shoulders: the torso is cut at the box
        auto cut = p;
        cut.box.height = 180;
        Appearance::regions(dress(size, cut, red, blue), cut, torso, legs);
        check(torso.br().y == 280, "torso is cut at the bottom of the box (bottom " + std::to_string(torso.br().y) + ")");
    }

    std::cout << "4. no keypoints: parts of the box for a standing person" << std::endl;
    {
        const auto p = person(500, 100, false, false, false, false);
        const cv::Mat frame = dress(size, p, red, blue);
        cv::Rect torso, legs;
        Appearance::regions(frame, p, torso, legs);
        check(torso == cv::Rect(550, 220, 100, 180), "torso is 20-50% of the box height, half its width");
        check(legs == cv::Rect(550, 430, 100, 210), "legs are 55-90% of the box height, half its width");
        Appearance::Signature s;
        check(Appearance::describe(frame, p, s) && !s.torso.empty() && !s.legs.empty(), "both regions described");

        // not a standing person: nothing to go by
        auto wide = p;
        wide.box = cv::Rect2f(300, 300, 400, 200);
        Appearance::regions(frame, wide, torso, legs);
        check(torso.area() == 0 && legs.area() == 0, "a wide box has no regions");
        check(!Appearance::describe(frame, wide, s) && s.empty(), "so nothing is described");

        // the region is clipped to the frame
        auto edge = p;
        edge.box = cv::Rect2f(-100, 600, 200, 600);
        Appearance::regions(frame, edge, torso, legs);
        check(cv::Rect(0, 0, size.width, size.height).contains(torso.tl()) && torso.br().x <= size.width && legs.br().y <= size.height,
              "regions stay inside the frame");
    }

    std::cout << "5. hips without shoulders: legs from the keypoints, torso from the box" << std::endl;
    {
        auto p = person(500, 100, false, true, false, true);
        cv::Rect torso, legs;
        Appearance::regions(dress(size, p, red, blue), p, torso, legs);
        check(legs.y == 420 && legs.br().y == 690, "legs from the hips to the ankles");
        check(torso.y == 220 && torso.br().y == 400, "torso falls back to the box (tall box)");
    }

    std::cout << "6. low confidence keypoints are not used" << std::endl;
    {
        auto p = person(500, 100, true, true, true, true);
        for (auto& c : p.confidence) c = 0.1f;
        cv::Rect torso, legs;
        Appearance::regions(dress(size, p, red, blue), p, torso, legs);
        check(torso == cv::Rect(550, 220, 100, 180) && legs == cv::Rect(550, 430, 100, 210), "the box fallback is used instead");
    }

    std::cout << "7. regions that are too small say nothing" << std::endl;
    {
        PersonDetection tiny = person(10, 10, false, false, false, false);
        tiny.box = cv::Rect2f(10, 10, 20, 40);
        Appearance::Signature s;
        check(!Appearance::describe(cv::Mat(200, 200, CV_8UC3, cv::Scalar(0, 0, 255)), tiny, s), "a distant person is not described");
    }

    std::cout << "8. a very large region is scaled down, the colors stay" << std::endl;
    {
        const cv::Size huge(4000, 3000);
        PersonDetection big = person(0, 0, true, true, true, true);
        big.box = cv::Rect2f(500, 100, 2000, 2800);
        auto set = [&](int k, float x, float y) { setPoint(big, k, x, y); };
        set(PersonDetection::LEFT_SHOULDER, 700, 500);
        set(PersonDetection::RIGHT_SHOULDER, 2300, 500);
        set(PersonDetection::LEFT_HIP, 900, 1700);
        set(PersonDetection::RIGHT_HIP, 2100, 1700);
        set(PersonDetection::LEFT_KNEE, 900, 2300);
        set(PersonDetection::RIGHT_KNEE, 2100, 2300);
        set(PersonDetection::LEFT_ANKLE, 900, 2800);
        set(PersonDetection::RIGHT_ANKLE, 2100, 2800);
        cv::Mat frame(huge, CV_8UC3, cv::Scalar(120, 120, 120));
        cv::rectangle(frame, cv::Rect(600, 450, 1800, 1300), red, cv::FILLED);
        cv::rectangle(frame, cv::Rect(800, 1700, 1400, 1200), blue, cv::FILLED);

        cv::Rect torso, legs;
        Appearance::regions(frame, big, torso, legs);
        check(torso.area() > 40000 && legs.area() > 40000, "both regions are over the scaling limit");
        Appearance::Signature s;
        check(Appearance::describe(frame, big, s), "described");
        check(std::abs(sum(s.torso) - 1) < 1e-3 && std::abs(sum(s.legs) - 1) < 1e-3, "histograms are normalized after scaling");

        // the same clothes on a small person look the same
        const auto small = person(500, 100, true, true, true, true);
        Appearance::Signature t;
        Appearance::describe(dress(size, small, red, blue), small, t);
        const float same = Appearance::compare(s, t);
        check(same > 0.95f, "same clothes small and large look alike (" + std::to_string(same) + ")");
    }

    std::cout << "9. comparing signatures" << std::endl;
    {
        const auto p = person(500, 100, true, true, true, true);
        Appearance::Signature a, b, c;
        Appearance::describe(dress(size, p, red, blue), p, a);
        Appearance::describe(dress(size, p, red, blue), p, b);
        Appearance::describe(dress(size, p, green, black), p, c);
        check(Appearance::compare(a, b) > 0.99f, "same clothes: " + std::to_string(Appearance::compare(a, b)));
        check(Appearance::compare(a, c) < 0.3f, "other clothes: " + std::to_string(Appearance::compare(a, c)));

        // only the regions both have count
        Appearance::Signature torsoOnly = a, legsOnly = a;
        torsoOnly.legs.clear();
        legsOnly.torso.clear();
        check(Appearance::compare(torsoOnly, a) > 0.99f, "torso only against both: compares the torsos");
        check(Appearance::compare(a, legsOnly) > 0.99f, "both against legs only: compares the legs");
        Appearance::Signature torsoOnlyOther = c;
        torsoOnlyOther.legs.clear();
        check(Appearance::compare(torsoOnlyOther, a) < 0.3f, "other torso, one side without legs: low (" + std::to_string(Appearance::compare(torsoOnlyOther, a)) + ")");
        // same shirt, other trousers: the torso counts more than the legs
        Appearance::Signature mixed = a;
        mixed.legs = c.legs;
        const float shirtSame = Appearance::compare(a, mixed);
        check(shirtSame > 0.55f && shirtSame < 0.7f, "same shirt other trousers is in between, torso counts 60% (" + std::to_string(shirtSame) + ")");
        check(Appearance::compare(torsoOnly, legsOnly) == -1, "no shared region: -1");
        check(Appearance::compare(Appearance::Signature(), a) == -1, "empty signature: -1");
    }

    std::cout << "10. learned views" << std::endl;
    {
        Appearance model;
        check(model.empty() && model.similarity(view(0)) == -1, "nothing known: similarity -1");
        model.learn(Appearance::Signature(), false);
        check(model.empty(), "an empty signature is not learned");

        model.learn(view(0), false);
        model.learn(view(0), false);
        check(model.size() == 1 && model.anchors() == 0, "a duplicate view is not added");
        model.learn(view(0), true);
        check(model.size() == 1 && model.anchors() == 1, "a duplicate seen with the face recognized becomes an anchor");
        model.learn(view(0), false);
        check(model.anchors() == 1, "and stays one");
        check(model.similarity(view(0)) > 0.99f && model.similarity(view(5)) < 0.01f, "similarity is the closest view");
        model.clear();
        check(model.empty(), "clear forgets everything");

        // 12 views are kept, the 13th pushes out the oldest
        for (size_t i = 0; i < 12; i++) model.learn(view(i), false);
        check(model.size() == 12, "12 views kept");
        model.learn(view(12), false);
        check(model.size() == 12 && model.similarity(view(0)) < 0.01f && model.similarity(view(1)) > 0.99f && model.similarity(view(12)) > 0.99f,
              "the oldest view was forgotten");

        // anchors survive while there are other views to forget
        model.clear();
        for (size_t i = 0; i < 3; i++) model.learn(view(i), true);
        for (size_t i = 3; i < 15; i++) model.learn(view(i), false);
        check(model.size() == 12 && model.anchors() == 3, "anchors are kept among recent views");
        check(model.similarity(view(0)) > 0.99f && model.similarity(view(2)) > 0.99f, "the anchor views are still known");
        check(model.similarity(view(3)) < 0.01f && model.similarity(view(14)) > 0.99f, "the oldest non-anchor went first");

        // more than 6 anchors: the oldest anchor goes
        model.clear();
        for (size_t i = 0; i < 7; i++) model.learn(view(i), true);
        check(model.size() == 6 && model.anchors() == 6, "at most 6 anchors");
        check(model.similarity(view(0)) < 0.01f && model.similarity(view(1)) > 0.99f && model.similarity(view(6)) > 0.99f,
              "the oldest anchor was forgotten");

        // all anchors and the set is full: a new non-anchor view still fits by pushing out an anchor
        model.clear();
        for (size_t i = 0; i < 6; i++) model.learn(view(i), true);
        for (size_t i = 6; i < 12; i++) model.learn(view(i), false);
        model.learn(view(12), true);
        check(model.size() == 12 && model.anchors() == 6, "anchors stay capped when a 7th arrives into a full set");
        check(model.similarity(view(0)) < 0.01f && model.similarity(view(6)) > 0.99f, "the oldest anchor went, the other views stayed");
    }

    std::cout << (failures ? "\nFAILED" : "\nall checks passed") << std::endl;
    return failures ? 1 : 0;
}
