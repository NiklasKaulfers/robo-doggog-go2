#include "Appearance.h"

#include <algorithm>
#include <cmath>
#include "opencv2/imgproc.hpp"

using namespace robo::sdk::ml;

// keypoints below this confidence are not used to place the body regions
static constexpr float KEYPOINT_MIN_CONFIDENCE = 0.2f;
// regions with fewer pixels are too noisy to describe anything
static constexpr int MIN_REGION_PIXELS = 150;
// regions larger than this are scaled down before the histogram, more is not needed
static constexpr int MAX_REGION_PIXELS = 40000;
// the torso region is narrowed so arms and background don't get in, and starts below
// the collar
static constexpr float TORSO_NARROW = 0.12f;
static constexpr float TORSO_SKIP_TOP = 0.1f;
// without hips the torso is taken to reach this far below the shoulders (times the
// shoulder width)
static constexpr float TORSO_LENGTH_FROM_SHOULDERS = 1.2f;
// the leg keypoints lie on the middle of the legs, widen the region to cover them
static constexpr float LEG_WIDEN = 0.3f;
// fallback regions when the keypoints are missing, as parts of the box, only for boxes
// that look like a whole standing person
static constexpr float FALLBACK_MIN_ASPECT = 1.3f;
static constexpr float FALLBACK_TORSO_TOP = 0.2f, FALLBACK_TORSO_BOTTOM = 0.5f;
static constexpr float FALLBACK_LEGS_TOP = 0.55f, FALLBACK_LEGS_BOTTOM = 0.9f;
static constexpr float FALLBACK_WIDTH = 0.5f;
// the torso counts more, legs are often similar between people (jeans)
static constexpr float TORSO_WEIGHT = 0.6f;
// the appearance model: how many views are kept, how many of them anchors, and how
// alike a new view must be to an existing one to be a pointless duplicate
static constexpr size_t MAX_TEMPLATES = 12;
static constexpr size_t MAX_ANCHORS = 6;
static constexpr float DUPLICATE_SIMILARITY = 0.97f;

// histogram layout: dark pixels, gray pixels by brightness, colored pixels by hue and
// saturation. Low saturation/brightness pixels have no reliable hue, so they get bins of
// their own instead of adding noise to the hue bins.
static constexpr int DARK_VALUE = 50;
static constexpr int GRAY_SATURATION = 50;
static constexpr int GRAY_BINS = 3;
static constexpr int HUE_BINS = 12;
static constexpr int SATURATION_BINS = 2;
static constexpr int HISTOGRAM_SIZE = 1 + GRAY_BINS + HUE_BINS * SATURATION_BINS;

static std::vector<float> histogram(const cv::Mat& bgr, cv::Rect region) {
    region &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (region.area() < MIN_REGION_PIXELS) return {};

    cv::Mat patch = bgr(region);
    if (region.area() > MAX_REGION_PIXELS) {
        const double scale = std::sqrt(static_cast<double>(MAX_REGION_PIXELS) / region.area());
        cv::resize(patch, patch, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);

    std::vector<float> hist(HISTOGRAM_SIZE, 0);
    for (int y = 0; y < hsv.rows; y++) {
        const auto* row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv.cols; x++) {
            const int h = row[x][0], s = row[x][1], v = row[x][2];
            if (v < DARK_VALUE) {
                hist[0]++;
            } else if (s < GRAY_SATURATION) {
                hist[1 + std::min(GRAY_BINS - 1, (v - DARK_VALUE) * GRAY_BINS / (256 - DARK_VALUE))]++;
            } else {
                const int satBin = std::min(SATURATION_BINS - 1, (s - GRAY_SATURATION) * SATURATION_BINS / (256 - GRAY_SATURATION));
                // the hue is split between its two nearest bins, so a color on a bin
                // border doesn't flip between bins with the lighting (hue is circular)
                const float position = h * HUE_BINS / 180.0f - 0.5f;
                const int lower = static_cast<int>(std::floor(position));
                const float upperWeight = position - lower;
                const int lowerBin = (lower + HUE_BINS) % HUE_BINS;
                const int upperBin = (lower + 1) % HUE_BINS;
                hist[1 + GRAY_BINS + lowerBin * SATURATION_BINS + satBin] += 1 - upperWeight;
                hist[1 + GRAY_BINS + upperBin * SATURATION_BINS + satBin] += upperWeight;
            }
        }
    }
    const float total = static_cast<float>(hsv.rows * hsv.cols);
    for (auto& bin : hist) bin /= total;
    return hist;
}

// Bhattacharyya coefficient, 1 for identical color distributions
static float overlap(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) sum += std::sqrt(a[i] * b[i]);
    return std::min(1.0f, sum);
}

static cv::Rect toRect(float x0, float y0, float x1, float y1) {
    return cv::Rect(cv::Point(std::lround(x0), std::lround(y0)), cv::Point(std::lround(x1), std::lround(y1)));
}

void Appearance::regions(const cv::Mat& frame, const PersonDetection& person, cv::Rect& torso, cv::Rect& legs) {
    torso = legs = cv::Rect();
    const auto has = [&](int k) { return person.has(k, KEYPOINT_MIN_CONFIDENCE); };
    const auto at = [&](int k) { return person.keypoints[k]; };

    const bool shoulders = has(PersonDetection::LEFT_SHOULDER) && has(PersonDetection::RIGHT_SHOULDER);
    const bool hips = has(PersonDetection::LEFT_HIP) && has(PersonDetection::RIGHT_HIP);
    const bool knees = has(PersonDetection::LEFT_KNEE) && has(PersonDetection::RIGHT_KNEE);
    const bool ankles = has(PersonDetection::LEFT_ANKLE) && has(PersonDetection::RIGHT_ANKLE);
    const bool wholeBody = person.box.height >= FALLBACK_MIN_ASPECT * person.box.width;

    const cv::Point2f ls = at(PersonDetection::LEFT_SHOULDER), rs = at(PersonDetection::RIGHT_SHOULDER);
    const cv::Point2f lh = at(PersonDetection::LEFT_HIP), rh = at(PersonDetection::RIGHT_HIP);

    // torso: between the shoulders and the hips, narrowed so the arms stay out
    if (shoulders) {
        const float shoulderY = (ls.y + rs.y) / 2;
        float x0, x1, bottom;
        if (hips) {
            x0 = std::min({ls.x, rs.x, lh.x, rh.x});
            x1 = std::max({ls.x, rs.x, lh.x, rh.x});
            bottom = (lh.y + rh.y) / 2;
        } else {
            x0 = std::min(ls.x, rs.x);
            x1 = std::max(ls.x, rs.x);
            bottom = std::min(shoulderY + TORSO_LENGTH_FROM_SHOULDERS * (x1 - x0), person.box.y + person.box.height);
        }
        const float narrow = (x1 - x0) * TORSO_NARROW;
        const float top = shoulderY + (bottom - shoulderY) * TORSO_SKIP_TOP;
        torso = toRect(x0 + narrow, top, x1 - narrow, bottom);
    } else if (wholeBody) {
        const float cx = person.box.x + person.box.width / 2;
        const float half = person.box.width * FALLBACK_WIDTH / 2;
        torso = toRect(cx - half, person.box.y + person.box.height * FALLBACK_TORSO_TOP,
                       cx + half, person.box.y + person.box.height * FALLBACK_TORSO_BOTTOM);
    }

    // legs: from the hips down to the ankles, or the knees when the feet are cut off
    if (hips && (knees || ankles)) {
        const cv::Point2f l = ankles ? at(PersonDetection::LEFT_ANKLE) : at(PersonDetection::LEFT_KNEE);
        const cv::Point2f r = ankles ? at(PersonDetection::RIGHT_ANKLE) : at(PersonDetection::RIGHT_KNEE);
        float x0 = std::min({lh.x, rh.x, l.x, r.x});
        float x1 = std::max({lh.x, rh.x, l.x, r.x});
        const float widen = (x1 - x0) * LEG_WIDEN;
        legs = toRect(x0 - widen, (lh.y + rh.y) / 2, x1 + widen, (l.y + r.y) / 2);
    } else if (!hips && !shoulders && wholeBody) {
        const float cx = person.box.x + person.box.width / 2;
        const float half = person.box.width * FALLBACK_WIDTH / 2;
        legs = toRect(cx - half, person.box.y + person.box.height * FALLBACK_LEGS_TOP,
                      cx + half, person.box.y + person.box.height * FALLBACK_LEGS_BOTTOM);
    }

    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    torso &= bounds;
    legs &= bounds;
}

bool Appearance::describe(const cv::Mat& frame, const PersonDetection& person, Signature& out) {
    out = Signature();
    if (frame.empty()) return false;
    cv::Rect torso, legs;
    regions(frame, person, torso, legs);
    if (torso.area() > 0) out.torso = histogram(frame, torso);
    if (legs.area() > 0) out.legs = histogram(frame, legs);
    return !out.empty();
}

float Appearance::compare(const Signature& a, const Signature& b) {
    // only the regions visible in both count, e.g. only the legs when standing close
    const bool torso = !a.torso.empty() && !b.torso.empty();
    const bool legs = !a.legs.empty() && !b.legs.empty();
    if (torso && legs) return TORSO_WEIGHT * overlap(a.torso, b.torso) + (1 - TORSO_WEIGHT) * overlap(a.legs, b.legs);
    if (torso) return overlap(a.torso, b.torso);
    if (legs) return overlap(a.legs, b.legs);
    return -1;
}

size_t Appearance::anchors() const {
    return std::count_if(templates.begin(), templates.end(), [](const Template& t) { return t.anchor; });
}

float Appearance::similarity(const Signature& signature) const {
    float best = -1;
    for (const auto& t : templates) best = std::max(best, compare(signature, t.signature));
    return best;
}

void Appearance::learn(const Signature& signature, bool anchor) {
    if (signature.empty()) return;

    // a view just like one we have adds nothing; it can still promote that one to an anchor
    for (auto& t : templates) {
        if (compare(signature, t.signature) >= DUPLICATE_SIMILARITY) {
            t.anchor = t.anchor || anchor;
            return;
        }
    }

    templates.push_back({signature, anchor});

    // make room: the oldest non-anchor goes first, anchors only when there are too many
    while (templates.size() > MAX_TEMPLATES || anchors() > MAX_ANCHORS) {
        const bool tooManyAnchors = anchors() > MAX_ANCHORS;
        auto victim = std::find_if(templates.begin(), templates.end(),
                                   [&](const Template& t) { return t.anchor == tooManyAnchors; });
        if (victim == templates.end()) victim = templates.begin();
        templates.erase(victim);
    }
}
