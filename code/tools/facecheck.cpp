// Checks the reference photos of the person to follow without the robot: loads them the
// way the app does, shows how well they agree with each other, and optionally runs the
// recognition over further images (e.g. debug frames saved by the app) to see which
// faces would be taken as the person. Needs only OpenCV, so it also runs on a laptop.
//
//   facecheck <photos-folder> [--res DIR] [--out DIR] [image-or-folder ...]
//
// --res DIR  folder with the .onnx models (default ./res)
// --out DIR  write every test image with the faces and similarities drawn in to DIR

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "../src/ml/FaceRecognizer.h"

using namespace robo::sdk::ml;

static bool isImage(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static const char* verdictName(FaceRecognizer::Verdict v) {
    switch (v) {
        case FaceRecognizer::Verdict::Match: return "MATCH";
        case FaceRecognizer::Verdict::Mismatch: return "mismatch";
        default: return "unsure";
    }
}

int main(int argc, char** argv) {
    std::string photos, resourceDir = "./res", outDir;
    std::vector<std::filesystem::path> tests;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--res" && i + 1 < argc) resourceDir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) outDir = argv[++i];
        else if (photos.empty()) photos = arg;
        else tests.emplace_back(arg);
    }
    if (photos.empty()) {
        std::cerr << "usage: facecheck <photos-folder> [--res DIR] [--out DIR] [image-or-folder ...]" << std::endl;
        return 1;
    }

    FaceRecognizer faces;
    if (!faces.init(resourceDir)) return 1;
    if (faces.enrollFolder(photos) == 0) {
        std::cerr << "no usable face in " << photos << std::endl;
        return 1;
    }

    // how the photos agree with each other: all of them should match each other, a
    // photo that matches none is probably someone else, a bad angle or very blurry
    const auto& refs = faces.referenceDetails();
    std::cout << "\nreference photos (face size in px, detector score):" << std::endl;
    for (const auto& ref : refs)
        std::cout << "  " << std::left << std::setw(30) << ref.file << std::right << std::setw(5) << static_cast<int>(ref.faceSize)
                  << " px  " << std::fixed << std::setprecision(2) << ref.quality << std::endl;

    if (refs.size() > 1) {
        std::cout << "\nsimilarity between the photos (>= " << FaceRecognizer::MATCH_THRESHOLD << " is a match):" << std::endl;
        std::cout << std::setw(30) << "";
        for (size_t j = 0; j < refs.size(); j++) std::cout << std::setw(6) << j + 1;
        std::cout << std::endl;
        for (size_t i = 0; i < refs.size(); i++) {
            std::cout << std::setw(2) << i + 1 << " " << std::left << std::setw(27) << refs[i].file.substr(0, 27) << std::right;
            for (size_t j = 0; j < refs.size(); j++) {
                if (i == j) std::cout << std::setw(6) << "-";
                else std::cout << std::setw(6) << std::fixed << std::setprecision(2) << FaceRecognizer::cosine(refs[i].feature, refs[j].feature);
            }
            std::cout << std::endl;
        }
    }

    // test images: every face with its similarity to the photos
    std::vector<std::filesystem::path> images;
    for (const auto& test : tests) {
        if (std::filesystem::is_directory(test)) {
            for (const auto& entry : std::filesystem::directory_iterator(test)) {
                if (isImage(entry.path())) images.push_back(entry.path());
            }
        } else {
            images.push_back(test);
        }
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) return 0;

    if (!outDir.empty()) std::filesystem::create_directories(outDir);
    std::cout << "\ntest images:" << std::endl;
    for (const auto& file : images) {
        cv::Mat image = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "  " << file << ": cannot read" << std::endl;
            continue;
        }
        const auto found = faces.observeAll(image);
        std::cout << "  " << file.filename().string() << ": " << found.size() << " face(s)" << std::endl;
        for (const auto& face : found) {
            std::cout << "      " << std::setw(4) << static_cast<int>(face.size) << " px at (" << static_cast<int>(face.box.x) << ", "
                      << static_cast<int>(face.box.y) << ")  similarity " << std::fixed << std::setprecision(2) << face.similarity
                      << "  " << verdictName(face.verdict) << std::endl;
            if (outDir.empty()) continue;
            const cv::Scalar color = face.verdict == FaceRecognizer::Verdict::Match ? cv::Scalar(0, 255, 0)
                                   : face.verdict == FaceRecognizer::Verdict::Mismatch ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 255);
            cv::rectangle(image, face.box, color, 2);
            std::ostringstream label;
            label << std::fixed << std::setprecision(2) << face.similarity;
            cv::putText(image, label.str(), cv::Point(face.box.x, std::max(12.0f, face.box.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (!outDir.empty()) cv::imwrite((std::filesystem::path(outDir) / file.filename()).string(), image);
    }
    return 0;
}
