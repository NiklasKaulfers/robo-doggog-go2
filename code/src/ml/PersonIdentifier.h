#ifndef FOLLOW_PERSONIDENTIFIER_H
#define FOLLOW_PERSONIDENTIFIER_H

#include <deque>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "movenet/Multipose.h"

namespace robo::sdk::ml {
// Recognizes one specific person. Their face (compared against the reference images)
// is what confirms who they are. Once confirmed, the colors of their clothes are learned
// from the robot's camera, so they can still be told apart from others while only seen
// from behind, which is most of the time when following someone.
class PersonIdentifier {

public:

    enum class Face {
        // no face visible, or too small to recognize
        Unknown,
        Match,
        Mismatch,
    };

private:

    // normalized color histograms of the upper body and the legs
    struct Signature {
        std::vector<float> upper;
        std::vector<float> lower;
    };

    // a face found in the current frame, in frame pixels
    struct FrameFace {
        cv::Rect2f box;
        // cosine similarity to the closest reference face
        float similarity;
    };

    cv::Ptr<cv::FaceDetectorYN> faceDetector;
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer;
    std::vector<cv::Mat> referenceFaces;
    std::vector<FrameFace> faces;
    // clothes of the person seen while their face was recognized, most recent last
    std::deque<Signature> clothes;

public:

    // loads the face models and the face in every image (jpg/png) in the folder.
    // Returns false if no face was found in any of the images.
    bool load(const std::string& folder);

    // finds and recognizes the faces in the detector's last frame
    void detectFaces(const movenet::Multipose& detector);
    // forgets the faces of the last frame, for frames where detectFaces is skipped
    void clearFaces();

    // whether the face of the detected person (if visible) is the reference person's;
    // similarity is set to the face similarity when a face was found
    Face face(const movenet::Multipose& detector, int detection, float* similarity = nullptr) const;

    // how much the detection's clothes look like the ones learned with learnClothing,
    // 0..1 (1 = same clothes). -1 if nothing learned yet or too little of the person is visible
    float clothing(const movenet::Multipose& detector, int detection) const;
    // remembers the clothes of the detection, call it only when their face matched
    void learnClothing(const movenet::Multipose& detector, int detection);

private:

    bool findFaces(const cv::Mat& image, cv::Mat& found);
    cv::Mat faceFeature(const cv::Mat& image, const cv::Mat& face);
    static bool signature(const movenet::Multipose& detector, int detection, Signature& out);
    static float compare(const Signature& a, const Signature& b);
};
}

#endif //FOLLOW_PERSONIDENTIFIER_H
