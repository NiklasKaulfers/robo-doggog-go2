#include "FaceRecognizer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace robo::sdk::ml;

// https://github.com/opencv/opencv_zoo, the 2022 YuNet also runs on OpenCV < 4.8
static constexpr const char* DETECTION_MODEL = "face_detection_yunet_2022mar.onnx";
static constexpr const char* RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx";
// detector score needed for a face; inside a head region a lower one is fine, the pose
// detector already says there is a head
static constexpr float DETECTION_SCORE = 0.7f;
static constexpr float NMS_THRESHOLD = 0.3f;
// head crops are scaled so their short side is at least this many pixels, YuNet misses
// faces that are tiny in its input, and at most this many to bound the cost
static constexpr int MIN_CROP_SIDE = 128;
static constexpr int MAX_CROP_SIDE = 320;
// large photos are scaled down first, the detector doesn't need more and is faster
static constexpr int MAX_PHOTO_SIZE = 1280;
// faces smaller than this in a reference photo are too blurry to serve as a reference
static constexpr float MIN_REFERENCE_FACE = 40;
// reference photos whose face is this dissimilar to all other photos are probably not
// the same person; they are kept, but a warning is printed
static constexpr float REFERENCE_OUTLIER = 0.25f;
// live-learned faces: how many are kept, and how different a new one has to be from
// those to be worth keeping (otherwise the newest one just replaces the near-identical)
static constexpr size_t MAX_LEARNED = 8;
static constexpr float LEARNED_NOVELTY = 0.92f;
// learning from the camera only happens on faces that are surely the person
static constexpr float LEARN_THRESHOLD = 0.45f;

bool FaceRecognizer::init(const std::string& resourceDir) {
    const auto detectionModel = (std::filesystem::path(resourceDir) / DETECTION_MODEL).string();
    const auto recognitionModel = (std::filesystem::path(resourceDir) / RECOGNITION_MODEL).string();
    try {
        detector = cv::FaceDetectorYN::create(detectionModel, "", cv::Size(320, 320), DETECTION_SCORE, NMS_THRESHOLD);
        recognizer = cv::FaceRecognizerSF::create(recognitionModel, "");
    } catch (const cv::Exception& e) {
        std::cerr << "[FaceRecognizer] unable to load the face models from " << resourceDir << ": " << e.what() << std::endl;
        return false;
    }
    return true;
}

int FaceRecognizer::enrollFolder(const std::string& folder) {
    std::vector<std::filesystem::path> files;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") files.push_back(entry.path());
    }
    if (error) {
        std::cerr << "[FaceRecognizer] could not open " << folder << ": " << error.message() << std::endl;
        return 0;
    }
    std::sort(files.begin(), files.end());

    int enrolled = 0;
    for (const auto& file : files) {
        // imread applies the EXIF orientation, so phone photos come out upright
        cv::Mat image = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[FaceRecognizer] could not read " << file << std::endl;
            continue;
        }
        const double scale = static_cast<double>(MAX_PHOTO_SIZE) / std::max(image.cols, image.rows);
        if (scale < 1) cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::Mat faces;
        if (!detect(image, faces) || faces.rows == 0) {
            std::cerr << "[FaceRecognizer] no face found in " << file << std::endl;
            continue;
        }

        // the largest face is taken as the person, photos of them should show them up front
        int best = 0;
        for (int i = 1; i < faces.rows; i++) {
            if (faces.at<float>(i, 2) * faces.at<float>(i, 3) > faces.at<float>(best, 2) * faces.at<float>(best, 3)) best = i;
        }
        const float size = std::min(faces.at<float>(best, 2), faces.at<float>(best, 3));
        if (size < MIN_REFERENCE_FACE) {
            std::cerr << "[FaceRecognizer] the face in " << file << " is too small (" << size
                      << " px), use a photo taken closer up" << std::endl;
            continue;
        }

        Reference ref;
        ref.file = file.filename().string();
        ref.faceSize = size;
        ref.quality = faces.at<float>(best, 14);
        ref.feature = feature(image, faces.row(best));
        std::cout << "[FaceRecognizer] enrolled " << ref.file << " (face " << static_cast<int>(size) << " px"
                  << (faces.rows > 1 ? ", largest of " + std::to_string(faces.rows) + " faces" : "") << ")" << std::endl;
        references.push_back(std::move(ref));
        enrolled++;
    }

    // the photos should all show the same person: warn about the ones that don't look like the rest
    if (references.size() > 1) {
        for (size_t i = 0; i < references.size(); i++) {
            float closest = -1;
            for (size_t j = 0; j < references.size(); j++) {
                if (i != j) closest = std::max(closest, cosine(references[i].feature, references[j].feature));
            }
            if (closest < REFERENCE_OUTLIER)
                std::cerr << "[FaceRecognizer] warning: the face in " << references[i].file
                          << " looks different from all other photos (similarity " << closest
                          << "), is it the same person?" << std::endl;
        }
    }
    return enrolled;
}

void FaceRecognizer::learn(const cv::Mat& feature, bool permanent) {
    if (feature.empty()) return;
    if (permanent) {
        Reference ref;
        ref.file = "camera";
        ref.feature = feature.clone();
        references.push_back(std::move(ref));
        return;
    }
    // not worth keeping if it looks just like one we have, or isn't surely the person
    if (similarity(feature) < LEARN_THRESHOLD) return;
    for (const auto& known : learned) {
        if (cosine(feature, known) >= LEARNED_NOVELTY) return;
    }
    learned.push_back(feature.clone());
    if (learned.size() > MAX_LEARNED) learned.pop_front();
}

FaceRecognizer::Observation FaceRecognizer::observe(const cv::Mat& frame, const cv::Rect2f& headRegion) {
    Observation none;
    if (frame.empty() || !detector) return none;

    // the crop is scaled so the face is big enough for the detector, but small heads
    // (far away people) are not worth it: nothing would be recognized anyway
    cv::Rect region(cv::Point(std::floor(headRegion.x), std::floor(headRegion.y)),
                    cv::Point(std::ceil(headRegion.x + headRegion.width), std::ceil(headRegion.y + headRegion.height)));
    region &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (region.width < MIN_FACE_SIZE || region.height < MIN_FACE_SIZE) return none;

    const int shortSide = std::min(region.width, region.height);
    const int longSide = std::max(region.width, region.height);
    float scale = 1;
    if (shortSide < MIN_CROP_SIDE) scale = static_cast<float>(MIN_CROP_SIDE) / shortSide;
    if (longSide * scale > MAX_CROP_SIDE) scale = static_cast<float>(MAX_CROP_SIDE) / longSide;

    cv::Mat crop;
    if (std::abs(scale - 1) < 0.01f) {
        crop = frame(region).clone();
        scale = 1;
    } else {
        cv::resize(frame(region), crop, cv::Size(), scale, scale, scale > 1 ? cv::INTER_LINEAR : cv::INTER_AREA);
    }

    cv::Mat faces;
    if (!detect(crop, faces) || faces.rows == 0) return none;

    // the surest face in the region, a head region is small so it is the only one normally
    int best = 0;
    for (int i = 1; i < faces.rows; i++) {
        if (faces.at<float>(i, 14) > faces.at<float>(best, 14)) best = i;
    }
    return describe(crop, faces.row(best), scale, cv::Point2f(region.x, region.y));
}

std::vector<FaceRecognizer::Observation> FaceRecognizer::observeAll(const cv::Mat& image) {
    std::vector<Observation> result;
    cv::Mat faces;
    if (image.empty() || !detect(image, faces)) return result;
    for (int i = 0; i < faces.rows; i++) result.push_back(describe(image, faces.row(i), 1, cv::Point2f(0, 0)));
    return result;
}

FaceRecognizer::Observation FaceRecognizer::describe(const cv::Mat& image, const cv::Mat& face, float scale, cv::Point2f offset) {
    Observation obs;
    obs.found = true;
    obs.box = cv::Rect2f(offset.x + face.at<float>(0, 0) / scale, offset.y + face.at<float>(0, 1) / scale,
                         face.at<float>(0, 2) / scale, face.at<float>(0, 3) / scale);
    obs.size = std::min(obs.box.width, obs.box.height);
    obs.quality = face.at<float>(0, 14);
    if (obs.size < MIN_FACE_SIZE) return obs;

    obs.feature = feature(image, face);
    obs.similarity = similarity(obs.feature);
    obs.verdict = verdict(obs.similarity, obs.size);
    return obs;
}

float FaceRecognizer::cosine(const cv::Mat& a, const cv::Mat& b) {
    const double denominator = cv::norm(a) * cv::norm(b);
    if (denominator <= 0) return 0;
    return static_cast<float>(a.dot(b) / denominator);
}

float FaceRecognizer::similarity(const cv::Mat& feature) const {
    float best = -1;
    // different photos show different angles and lighting, the closest one counts
    for (const auto& ref : references) best = std::max(best, cosine(feature, ref.feature));
    for (const auto& known : learned) best = std::max(best, cosine(feature, known));
    return best;
}

FaceRecognizer::Verdict FaceRecognizer::verdict(float similarity, float faceSize) {
    if (similarity < 0 || faceSize < MIN_FACE_SIZE) return Verdict::Unknown;
    if (similarity >= MATCH_THRESHOLD) return Verdict::Match;
    if (similarity < MISMATCH_THRESHOLD && faceSize >= MIN_MISMATCH_FACE_SIZE) return Verdict::Mismatch;
    return Verdict::Unknown;
}

bool FaceRecognizer::detect(const cv::Mat& image, cv::Mat& faces) {
    try {
        detector->setInputSize(image.size());
        detector->detect(image, faces);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[FaceRecognizer] face detection failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat FaceRecognizer::feature(const cv::Mat& image, const cv::Mat& face) {
    cv::Mat aligned, result;
    recognizer->alignCrop(image, face, aligned);
    recognizer->feature(aligned, result);
    // feature() reuses its output buffer, keep a copy
    return result.clone();
}
