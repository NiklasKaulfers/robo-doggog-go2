#include "PersonIdentifier.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace robo::sdk::ml;
using namespace robo::sdk::ml::movenet;

// https://github.com/opencv/opencv_zoo, the 2022 YuNet also runs on OpenCV < 4.8
static constexpr const char* FACE_DETECTION_MODEL = "./res/face_detection_yunet_2022mar.onnx";
static constexpr const char* FACE_RECOGNITION_MODEL = "./res/face_recognition_sface_2021dec.onnx";
static constexpr float FACE_DETECTION_CONFIDENCE = 0.8f;
// cosine similarity of SFace features: OpenCV's recommended threshold for the same
// person, and below which a face is taken as clearly someone else (in between: unsure)
static constexpr float FACE_MATCH = 0.363f;
static constexpr float FACE_MISMATCH = 0.2f;
// smaller faces (pixels) are too blurry to recognize, the person is too far away
static constexpr float MIN_FACE_SIZE = 24;
// large photos are scaled down first, the detector doesn't need more and is faster
static constexpr int MAX_REFERENCE_SIZE = 1280;
// without a confident nose, a face belongs to a person if it is in the top of their box
static constexpr float HEAD_PART_OF_BOX = 0.4f;

// keypoints below this confidence are not used to place the body regions
static constexpr float KEYPOINT_MIN_CONFIDENCE = 0.2f;
// regions smaller than this (in pixels of the 256x256 model input) are too noisy
static constexpr int MIN_REGION_PIXELS = 60;
// the leg keypoints lie on the middle of the legs, widen the region to cover them
static constexpr float LEG_WIDEN = 0.3f;
// the upper body counts a bit more, legs are often similar (jeans)
static constexpr float UPPER_WEIGHT = 0.6f;
// how many views of the clothes are remembered (front, back, different lighting)
static constexpr size_t MAX_CLOTHES = 10;

// histogram layout: dark pixels, gray pixels by brightness, colored pixels by hue and
// saturation. Low saturation/brightness pixels have no reliable hue, so they get bins of
// their own instead of adding noise to the hue bins.
static constexpr int DARK_VALUE = 50;
static constexpr int GRAY_SATURATION = 50;
static constexpr int GRAY_BINS = 3;
static constexpr int HUE_BINS = 12;
static constexpr int SATURATION_BINS = 2;
static constexpr int HISTOGRAM_SIZE = 1 + GRAY_BINS + HUE_BINS * SATURATION_BINS;

// Movenet's keypoints are stored as (y, x, confidence), so Keypoint::x is the row and
// Keypoint::y the column of the point
static float column(const Keypoint& k) { return k.y; }
static float row(const Keypoint& k) { return k.x; }

static std::vector<float> histogram(const cv::Mat& rgb, cv::Rect region) {
    region &= cv::Rect(0, 0, rgb.cols, rgb.rows);
    if (region.area() < MIN_REGION_PIXELS) return {};

    cv::Mat hsv;
    cv::cvtColor(rgb(region), hsv, cv::COLOR_RGB2HSV);

    std::vector<float> hist(HISTOGRAM_SIZE, 0);
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            const auto& p = hsv.at<cv::Vec3b>(y, x);
            const int h = p[0], s = p[1], v = p[2];
            int bin;
            if (v < DARK_VALUE) {
                bin = 0;
            } else if (s < GRAY_SATURATION) {
                bin = 1 + std::min(GRAY_BINS - 1, (v - DARK_VALUE) * GRAY_BINS / (256 - DARK_VALUE));
            } else {
                const int hueBin = std::min(HUE_BINS - 1, h * HUE_BINS / 180);
                const int satBin = std::min(SATURATION_BINS - 1, (s - GRAY_SATURATION) * SATURATION_BINS / (256 - GRAY_SATURATION));
                bin = 1 + GRAY_BINS + hueBin * SATURATION_BINS + satBin;
            }
            hist[bin]++;
        }
    }
    const float total = static_cast<float>(hsv.rows * hsv.cols);
    for (auto& b : hist) b /= total;
    return hist;
}

// Bhattacharyya coefficient, 1 for identical color distributions
static float overlap(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++) sum += std::sqrt(a[i] * b[i]);
    return sum;
}

bool PersonIdentifier::load(const std::string& folder) {
    try {
        faceDetector = cv::FaceDetectorYN::create(FACE_DETECTION_MODEL, "", cv::Size(320, 320), FACE_DETECTION_CONFIDENCE);
        faceRecognizer = cv::FaceRecognizerSF::create(FACE_RECOGNITION_MODEL, "");
    } catch (const cv::Exception& e) {
        std::cerr << "[PersonIdentifier] unable to load the face models: " << e.what() << std::endl;
        return false;
    }

    referenceFaces.clear();
    clothes.clear();

    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[PersonIdentifier] could not read " << entry.path() << std::endl;
            continue;
        }
        const double scale = static_cast<double>(MAX_REFERENCE_SIZE) / std::max(image.cols, image.rows);
        if (scale < 1) cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::Mat found;
        if (!findFaces(image, found) || found.rows == 0) {
            std::cerr << "[PersonIdentifier] no face found in " << entry.path() << std::endl;
            continue;
        }

        // the largest face is taken as the reference person
        int best = 0;
        for (int i = 1; i < found.rows; i++) {
            if (found.at<float>(i, 2) * found.at<float>(i, 3) > found.at<float>(best, 2) * found.at<float>(best, 3)) best = i;
        }
        if (found.rows > 1)
            std::cout << "[PersonIdentifier] " << found.rows << " faces in " << entry.path()
                      << ", using the largest one" << std::endl;
        referenceFaces.push_back(faceFeature(image, found.row(best)));
    }

    if (error) {
        std::cerr << "[PersonIdentifier] could not open " << folder << ": " << error.message() << std::endl;
        return false;
    }
    std::cout << "[PersonIdentifier] loaded " << referenceFaces.size() << " reference face(s) from " << folder << std::endl;
    return !referenceFaces.empty();
}

void PersonIdentifier::detectFaces(const movenet::Multipose& detector) {
    faces.clear();
    const cv::Mat& frame = detector.frame();
    cv::Mat found;
    if (frame.empty() || !findFaces(frame, found)) return;

    for (int i = 0; i < found.rows; i++) {
        const cv::Rect2f box(found.at<float>(i, 0), found.at<float>(i, 1), found.at<float>(i, 2), found.at<float>(i, 3));
        if (std::min(box.width, box.height) < MIN_FACE_SIZE) continue;

        // different images show different angles/lighting, the closest one counts
        const cv::Mat feature = faceFeature(frame, found.row(i));
        float best = -1;
        for (const auto& ref : referenceFaces)
            best = std::max(best, static_cast<float>(faceRecognizer->match(feature, ref, cv::FaceRecognizerSF::FR_COSINE)));
        faces.push_back({box, best});
    }
}

void PersonIdentifier::clearFaces() {
    faces.clear();
}

PersonIdentifier::Face PersonIdentifier::face(const movenet::Multipose& detector, int detection, float* similarity) const {
    const auto box = detector.getBox(detection);
    const cv::Point2f topLeft = detector.toFrame(box.xmin, box.ymin);
    const cv::Point2f bottomRight = detector.toFrame(box.xmax, box.ymax);
    const cv::Rect2f head(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, (bottomRight.y - topLeft.y) * HEAD_PART_OF_BOX);

    const auto nose = detector.getKeypoint(detection, Movenet::KEYPOINT_NOSE);
    const bool hasNose = nose.confidence >= KEYPOINT_MIN_CONFIDENCE;
    const cv::Point2f nosePoint = detector.toFrame(column(nose), row(nose));

    // the largest face that belongs to this person: around their nose, or else in the
    // top of their box
    const FrameFace* best = nullptr;
    for (const auto& f : faces) {
        const cv::Point2f center(f.box.x + f.box.width / 2, f.box.y + f.box.height / 2);
        const bool belongs = hasNose ? f.box.contains(nosePoint) : head.contains(center);
        if (belongs && (!best || f.box.area() > best->box.area())) best = &f;
    }
    if (!best) return Face::Unknown;

    if (similarity) *similarity = best->similarity;
    if (best->similarity >= FACE_MATCH) return Face::Match;
    if (best->similarity < FACE_MISMATCH) return Face::Mismatch;
    return Face::Unknown;
}

float PersonIdentifier::clothing(const movenet::Multipose& detector, int detection) const {
    if (clothes.empty()) return -1;
    Signature sig;
    if (!signature(detector, detection, sig)) return -1;

    float best = -1;
    for (const auto& known : clothes) best = std::max(best, compare(sig, known));
    return best;
}

void PersonIdentifier::learnClothing(const movenet::Multipose& detector, int detection) {
    Signature sig;
    if (!signature(detector, detection, sig)) return;
    clothes.push_back(std::move(sig));
    if (clothes.size() > MAX_CLOTHES) clothes.pop_front();
}

bool PersonIdentifier::findFaces(const cv::Mat& image, cv::Mat& found) {
    try {
        faceDetector->setInputSize(image.size());
        faceDetector->detect(image, found);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[PersonIdentifier] face detection failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat PersonIdentifier::faceFeature(const cv::Mat& image, const cv::Mat& face) {
    cv::Mat aligned, feature;
    faceRecognizer->alignCrop(image, face, aligned);
    faceRecognizer->feature(aligned, feature);
    // feature() reuses its output buffer, keep a copy
    return feature.clone();
}

bool PersonIdentifier::signature(const movenet::Multipose& detector, int detection, Signature& out) {
    const cv::Mat& image = detector.inputImage();
    if (image.empty()) return false;
    const float w = static_cast<float>(image.cols);
    const float h = static_cast<float>(image.rows);

    auto point = [&](int idx, Keypoint& k) {
        k = detector.getKeypoint(detection, idx);
        return k.confidence >= KEYPOINT_MIN_CONFIDENCE;
    };

    Keypoint ls, rs, lh, rh, lk, rk, la, ra;
    const bool shoulders = point(Movenet::KEYPOINT_LEFT_SHOULDER, ls) && point(Movenet::KEYPOINT_RIGHT_SHOULDER, rs);
    const bool hips = point(Movenet::KEYPOINT_LEFT_HIP, lh) && point(Movenet::KEYPOINT_RIGHT_HIP, rh);
    const bool knees = point(Movenet::KEYPOINT_LEFT_KNEE, lk) && point(Movenet::KEYPOINT_RIGHT_KNEE, rk);
    const bool ankles = point(Movenet::KEYPOINT_LEFT_ANKLE, la) && point(Movenet::KEYPOINT_RIGHT_ANKLE, ra);

    // upper body: the box between shoulders and hips
    if (shoulders && hips) {
        const float x0 = std::min({column(ls), column(rs), column(lh), column(rh)});
        const float x1 = std::max({column(ls), column(rs), column(lh), column(rh)});
        const float y0 = (row(ls) + row(rs)) / 2;
        const float y1 = (row(lh) + row(rh)) / 2;
        out.upper = histogram(image, cv::Rect(cv::Point(x0 * w, y0 * h), cv::Point(x1 * w, y1 * h)));
    }

    // legs: from the hips down to the ankles, or the knees when the feet are cut off
    if (hips && (knees || ankles)) {
        const Keypoint& l = ankles ? la : lk;
        const Keypoint& r = ankles ? ra : rk;
        float x0 = std::min({column(lh), column(rh), column(l), column(r)});
        float x1 = std::max({column(lh), column(rh), column(l), column(r)});
        const float widen = (x1 - x0) * LEG_WIDEN;
        x0 -= widen;
        x1 += widen;
        const float y0 = (row(lh) + row(rh)) / 2;
        const float y1 = (row(l) + row(r)) / 2;
        out.lower = histogram(image, cv::Rect(cv::Point(x0 * w, y0 * h), cv::Point(x1 * w, y1 * h)));
    }

    return !out.upper.empty() || !out.lower.empty();
}

float PersonIdentifier::compare(const Signature& a, const Signature& b) {
    // only the regions visible in both count, e.g. only the legs when standing close
    const bool upper = !a.upper.empty() && !b.upper.empty();
    const bool lower = !a.lower.empty() && !b.lower.empty();
    if (upper && lower) return UPPER_WEIGHT * overlap(a.upper, b.upper) + (1 - UPPER_WEIGHT) * overlap(a.lower, b.lower);
    if (upper) return overlap(a.upper, b.upper);
    if (lower) return overlap(a.lower, b.lower);
    return -1;
}
