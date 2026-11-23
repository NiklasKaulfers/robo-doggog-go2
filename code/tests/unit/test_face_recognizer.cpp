// FaceRecognizer logic that needs no models: similarity, verdicts and the gallery
#include <vector>
#include "doctest/doctest.h"
#include "ml/FaceRecognizer.h"

using robo::sdk::ml::FaceRecognizer;

// a 1 x n float row like an SFace embedding (the Mat_ comma initializer is gone in OpenCV 5)
static cv::Mat feature(const std::vector<float>& values) {
    return cv::Mat(1, static_cast<int>(values.size()), CV_32F, const_cast<float*>(values.data())).clone();
}

TEST_CASE("cosine similarity of a feature with itself is 1") {
    const cv::Mat f = feature({0.1f, -0.4f, 0.7f, 0.2f});
    CHECK(FaceRecognizer::cosine(f, f) == doctest::Approx(1.0f));
}

TEST_CASE("a clearly different face is a mismatch, also when its similarity is negative") {
    // SFace similarities of different people are often around 0, sometimes below. The
    // further below the mismatch threshold, the surer it is someone else
    CHECK(FaceRecognizer::verdict(0.10f, 60) == FaceRecognizer::Verdict::Mismatch);
    CHECK(FaceRecognizer::verdict(-0.05f, 60) == FaceRecognizer::Verdict::Mismatch);
    CHECK(FaceRecognizer::verdict(-0.5f, 60) == FaceRecognizer::Verdict::Mismatch);
}

// ---- helpers for the tests below ----

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using Verdict = FaceRecognizer::Verdict;

namespace {

constexpr int DIM = 32;

// unit vector along one axis
cv::Mat axis(int i) {
    std::vector<float> v(DIM, 0.0f);
    v[i] = 1.0f;
    return feature(v);
}

// normalize(base + axis(i)): cosine about 0.7 with base, about 0.5 with another such vector
cv::Mat variant(const cv::Mat& base, int i) {
    cv::Mat sum = base + axis(i);
    return sum / cv::norm(sum);
}

// a unique empty temporary folder, removed again on destruction
struct TempDir {
    std::filesystem::path path;
    TempDir() {
        static int counter = 0;
        path = std::filesystem::temp_directory_path() /
               ("face_recognizer_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

} // namespace

// ---- cosine ----

TEST_CASE("cosine similarity of opposite features is minus 1") {
    const cv::Mat f = feature({0.3f, -0.2f, 0.9f});
    const cv::Mat g = -f;
    CHECK(FaceRecognizer::cosine(f, g) == doctest::Approx(-1.0f));
}

TEST_CASE("cosine similarity of orthogonal features is 0") {
    CHECK(FaceRecognizer::cosine(axis(0), axis(1)) == doctest::Approx(0.0f).epsilon(1e-6));
}

TEST_CASE("cosine similarity ignores the length of the features") {
    const cv::Mat f = feature({0.3f, -0.2f, 0.9f, 0.1f});
    const cv::Mat scaled = f * 37.5;
    CHECK(FaceRecognizer::cosine(f, scaled) == doctest::Approx(1.0f));
    CHECK(FaceRecognizer::cosine(scaled, f) == doctest::Approx(1.0f));
}

TEST_CASE("cosine similarity with a zero feature is 0 and not NaN") {
    const cv::Mat zero = feature({0.0f, 0.0f, 0.0f});
    const cv::Mat f = feature({1.0f, 2.0f, 3.0f});
    const float a = FaceRecognizer::cosine(zero, f);
    const float b = FaceRecognizer::cosine(f, zero);
    const float c = FaceRecognizer::cosine(zero, zero);
    CHECK_FALSE(std::isnan(a));
    CHECK_FALSE(std::isnan(b));
    CHECK_FALSE(std::isnan(c));
    CHECK(a == 0.0f);
    CHECK(b == 0.0f);
    CHECK(c == 0.0f);
}

TEST_CASE("cosine similarity is symmetric for 128 dimensional features") {
    std::vector<float> a(128), b(128);
    // deterministic pseudo random values, no <random> seeding to depend on
    unsigned state = 12345;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / static_cast<float>(1u << 24) - 0.5f;
    };
    for (int i = 0; i < 128; i++) {
        a[i] = next();
        b[i] = next();
    }
    const cv::Mat fa = feature(a), fb = feature(b);
    const float ab = FaceRecognizer::cosine(fa, fb);
    CHECK(ab == doctest::Approx(FaceRecognizer::cosine(fb, fa)));
    CHECK(ab >= -1.0f);
    CHECK(ab <= 1.0f);
}

// ---- verdict ----

TEST_CASE("verdict follows the threshold table") {
    struct Row {
        float similarity;
        float size;
        Verdict expected;
    };
    const Row rows[] = {
        {0.363f, 20.0f, Verdict::Match},     // exactly at the match threshold on the smallest usable face
        {0.3629f, 60.0f, Verdict::Unknown},  // just below the match threshold
        {0.2f, 60.0f, Verdict::Unknown},     // between the thresholds
        {0.1499f, 32.0f, Verdict::Mismatch}, // just below the mismatch threshold on a large enough face
        {0.1499f, 31.9f, Verdict::Unknown},  // a small face is never a mismatch
        {0.9f, 19.9f, Verdict::Unknown},     // too small for anything
        {0.9f, 20.0f, Verdict::Match},       // smallest face that can match
    };
    for (const auto& row : rows) {
        CAPTURE(row.similarity);
        CAPTURE(row.size);
        CHECK(FaceRecognizer::verdict(row.similarity, row.size) == row.expected);
    }
}

// ---- gallery ----

TEST_CASE("an empty gallery has no references and similarity minus 1") {
    FaceRecognizer recognizer;
    CHECK(recognizer.gallerySize() == 0);
    CHECK_FALSE(recognizer.hasReferences());
    CHECK(recognizer.referenceDetails().empty());
    CHECK(recognizer.similarity(axis(0)) == -1.0f);
}

TEST_CASE("learning an empty feature is ignored") {
    FaceRecognizer recognizer;
    recognizer.learn(cv::Mat(), true);
    recognizer.learn(cv::Mat(), false);
    CHECK(recognizer.gallerySize() == 0);
    CHECK_FALSE(recognizer.hasReferences());
}

TEST_CASE("a permanent face becomes a camera reference") {
    FaceRecognizer recognizer;
    const cv::Mat f = axis(3);
    recognizer.learn(f, true);
    CHECK(recognizer.gallerySize() == 1);
    CHECK(recognizer.hasReferences());
    REQUIRE(recognizer.referenceDetails().size() == 1);
    CHECK(recognizer.referenceDetails()[0].file == "camera");
    CHECK(recognizer.similarity(f) == doctest::Approx(1.0f));
}

TEST_CASE("the gallery keeps a copy of a learned feature") {
    FaceRecognizer recognizer;
    cv::Mat f = axis(3);
    const cv::Mat original = f.clone();
    recognizer.learn(f, true);
    // overwriting the caller's buffer (the detector reuses its output) must not touch the gallery
    f.setTo(0);
    f.at<float>(0, 5) = 1.0f;
    CHECK(recognizer.similarity(original) == doctest::Approx(1.0f));
    CHECK(recognizer.similarity(f) == doctest::Approx(0.0f).epsilon(1e-6));
}

TEST_CASE("a live face is not learned while the gallery is empty") {
    FaceRecognizer recognizer;
    recognizer.learn(axis(0), false);
    CHECK(recognizer.gallerySize() == 0);
    CHECK(recognizer.similarity(axis(0)) == -1.0f);
}

TEST_CASE("a live face is learned only when it is similar enough to the person") {
    FaceRecognizer recognizer;
    const cv::Mat reference = axis(0);
    recognizer.learn(reference, true);

    // cosine 0.7 with the reference, above the learn threshold
    const cv::Mat similar = variant(reference, 1);
    REQUIRE(FaceRecognizer::cosine(similar, reference) >= 0.45f);
    recognizer.learn(similar, false);
    CHECK(recognizer.gallerySize() == 2);

    // cosine 0.3 with the reference and 0.21 with the learned one, below the learn threshold
    cv::Mat far = reference * 0.3 + axis(2) * std::sqrt(1.0 - 0.09);
    REQUIRE(FaceRecognizer::cosine(far, reference) < 0.45f);
    recognizer.learn(far, false);
    CHECK(recognizer.gallerySize() == 2);
}

TEST_CASE("a near duplicate of a learned face is not learned again") {
    FaceRecognizer recognizer;
    recognizer.learn(axis(0), true);
    const cv::Mat first = variant(axis(0), 1);
    recognizer.learn(first, false);
    REQUIRE(recognizer.gallerySize() == 2);

    // same direction with a small change: cosine with the learned face is above 0.92
    cv::Mat duplicate = first + axis(5) * 0.1;
    REQUIRE(FaceRecognizer::cosine(duplicate, first) >= 0.92f);
    recognizer.learn(duplicate, false);
    CHECK(recognizer.gallerySize() == 2);
}

TEST_CASE("at most 8 learned faces are kept and the oldest goes first") {
    FaceRecognizer recognizer;
    const cv::Mat reference = axis(0);
    recognizer.learn(reference, true);

    std::vector<cv::Mat> learned;
    for (int i = 0; i < 9; i++) {
        learned.push_back(variant(reference, i + 1));
        recognizer.learn(learned.back(), false);
    }
    // 1 reference and the newest 8 learned
    CHECK(recognizer.gallerySize() == 9);
    CHECK(recognizer.similarity(learned[0]) < 0.99f);
    CHECK(recognizer.similarity(learned[1]) == doctest::Approx(1.0f));
    CHECK(recognizer.similarity(learned[8]) == doctest::Approx(1.0f));
}

TEST_CASE("permanent faces are never dropped by learning") {
    FaceRecognizer recognizer;
    const cv::Mat p0 = axis(0), p1 = axis(1), p2 = axis(2);
    recognizer.learn(p0, true);
    recognizer.learn(p1, true);
    recognizer.learn(p2, true);
    for (int i = 0; i < 20; i++) recognizer.learn(variant(p0, 3 + i), false);

    CHECK(recognizer.referenceDetails().size() == 3);
    CHECK(recognizer.gallerySize() == 3 + 8);
    CHECK(recognizer.similarity(p0) == doctest::Approx(1.0f));
    CHECK(recognizer.similarity(p1) == doctest::Approx(1.0f));
    CHECK(recognizer.similarity(p2) == doctest::Approx(1.0f));
}

TEST_CASE("similarity is the best match over references and learned faces") {
    FaceRecognizer recognizer;
    const cv::Mat reference = axis(0);
    recognizer.learn(reference, true);
    const cv::Mat learned = variant(reference, 1);
    recognizer.learn(learned, false);
    REQUIRE(recognizer.gallerySize() == 2);

    const cv::Mat query = variant(learned, 2);
    const float expected = std::max(FaceRecognizer::cosine(query, reference), FaceRecognizer::cosine(query, learned));
    CHECK(recognizer.similarity(query) == doctest::Approx(expected));
    // the learned face is the closer one here, so the reference alone would score lower
    CHECK(recognizer.similarity(query) > FaceRecognizer::cosine(query, reference));
    CHECK(recognizer.similarity(reference) == doctest::Approx(1.0f));
    CHECK(recognizer.similarity(learned) == doctest::Approx(1.0f));
}

// ---- observe ----

TEST_CASE("an observation starts out as nothing found") {
    const FaceRecognizer::Observation obs;
    CHECK_FALSE(obs.found);
    CHECK_FALSE(obs.compared);
    CHECK(obs.similarity == -1.0f);
    CHECK(obs.verdict == Verdict::Unknown);
    CHECK(obs.quality == 0.0f);
    CHECK(obs.size == 0.0f);
    CHECK(obs.feature.empty());
}

TEST_CASE("a recognizer without models finds nothing in any region") {
    FaceRecognizer recognizer;
    recognizer.learn(axis(0), true);
    const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(90, 120, 150));
    const cv::Rect2f regions[] = {
        {100, 100, 200, 200}, {0, 0, 640, 480}, {600, 440, 200, 200}, {10, 10, 5, 5}, {-50, -50, 100, 100},
    };
    for (const auto& region : regions) {
        CAPTURE(region.x);
        CAPTURE(region.y);
        const auto obs = recognizer.observe(frame, region);
        CHECK_FALSE(obs.found);
        CHECK_FALSE(obs.compared);
        CHECK(obs.similarity == -1.0f);
        CHECK(obs.verdict == Verdict::Unknown);
    }
}

TEST_CASE("an empty frame finds nothing") {
    FaceRecognizer recognizer;
    const auto obs = recognizer.observe(cv::Mat(), cv::Rect2f(0, 0, 100, 100));
    CHECK_FALSE(obs.found);
    CHECK_FALSE(obs.compared);
    CHECK(obs.similarity == -1.0f);
    CHECK(obs.verdict == Verdict::Unknown);
}

// ---- init and enrollFolder ----

TEST_CASE("init fails when the resource directory does not exist") {
    FaceRecognizer recognizer;
    const auto missing = std::filesystem::temp_directory_path() / "face_recognizer_no_such_resources_dir";
    std::filesystem::remove_all(missing);
    CHECK_FALSE(recognizer.init(missing.string()));
}

TEST_CASE("enrolling a folder that does not exist gives 0") {
    FaceRecognizer recognizer;
    const auto missing = std::filesystem::temp_directory_path() / "face_recognizer_no_such_photo_dir";
    std::filesystem::remove_all(missing);
    CHECK(recognizer.enrollFolder(missing.string()) == 0);
    CHECK_FALSE(recognizer.hasReferences());
}

TEST_CASE("enrolling an empty folder gives 0") {
    TempDir dir;
    FaceRecognizer recognizer;
    CHECK(recognizer.enrollFolder(dir.path.string()) == 0);
    CHECK_FALSE(recognizer.hasReferences());
    CHECK(recognizer.gallerySize() == 0);
}

TEST_CASE("enrolling a folder without images gives 0") {
    TempDir dir;
    std::ofstream(dir.path / "notes.txt") << "not a photo";
    std::ofstream(dir.path / "data.json") << "{}";
    FaceRecognizer recognizer;
    CHECK(recognizer.enrollFolder(dir.path.string()) == 0);
    CHECK_FALSE(recognizer.hasReferences());
    CHECK(recognizer.gallerySize() == 0);
}

// ---- the models shipped in res/ ----

TEST_CASE("the face models in res load") {
    // the app stops right at the start without them. Runs with code/ as working directory
    // (ctest does that); detection itself needs an OpenCV that runs YuNet 2022mar (4.x)
    FaceRecognizer recognizer;
    CHECK(recognizer.init("./res"));
}
