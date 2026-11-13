#ifndef FOLLOW_FACERECOGNIZER_H
#define FOLLOW_FACERECOGNIZER_H

#include <deque>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"

namespace robo::sdk::ml {

// Recognizes the face of one person. The gallery of known faces comes from reference
// photos (enrollFolder) and/or from the live camera (learn), and every face seen is
// compared against the closest one in the gallery.
//
// Faces are looked for inside a given head region of the frame instead of the whole
// frame: the pose detector already knows where every head is, a small full-resolution
// crop finds far away faces better than a downscaled full frame, and it is cheap.
class FaceRecognizer {

public:

    enum class Verdict {
        // no face, too small, or similarity between the thresholds
        Unknown,
        Match,
        Mismatch,
    };

    // what was found when looking for a face in a head region
    struct Observation {
        bool found = false;
        // in frame pixels
        cv::Rect2f box;
        // smaller side of the box, in frame pixels: small faces are unreliable
        float size = 0;
        // detector score, 0..1
        float quality = 0;
        // cosine similarity to the closest gallery face, -1 if the gallery is empty
        float similarity = -1;
        Verdict verdict = Verdict::Unknown;
        // the SFace embedding, for learning
        cv::Mat feature;
    };

    // a reference photo as loaded by enrollFolder, for the offline check tool
    struct Reference {
        std::string file;
        float faceSize = 0;
        float quality = 0;
        cv::Mat feature;
    };

    // cosine similarity thresholds. OpenCV recommends 0.363 for SFace; between the two
    // the face is not decided
    static constexpr float MATCH_THRESHOLD = 0.363f;
    static constexpr float MISMATCH_THRESHOLD = 0.15f;
    // a match this good is taken as certain
    static constexpr float STRONG_MATCH = 0.5f;

    // smaller faces (frame pixels) are not recognized
    static constexpr float MIN_FACE_SIZE = 20;
    // a mismatch is only trusted on a face at least this large
    static constexpr float MIN_MISMATCH_FACE_SIZE = 32;
    // faces at least this large and sharp are worth learning from
    static constexpr float MIN_LEARN_FACE_SIZE = 40;
    static constexpr float MIN_LEARN_QUALITY = 0.9f;

    // loads the models, resourceDir is the folder holding the .onnx files
    bool init(const std::string& resourceDir = "./res");

    // enrolls the largest face of every jpg/png in the folder; returns how many
    int enrollFolder(const std::string& folder);
    // adds a face seen on the camera. permanent faces are kept forever (the person's
    // identity when there are no photos), learned ones are a rolling set of recent views
    void learn(const cv::Mat& feature, bool permanent);

    bool hasReferences() const { return !references.empty(); }
    size_t gallerySize() const { return references.size() + learned.size(); }
    const std::vector<Reference>& referenceDetails() const { return references; }

    // looks for a face inside the head region (frame pixels) and compares it to the gallery
    Observation observe(const cv::Mat& frame, const cv::Rect2f& headRegion);

    // all faces in an image with their similarity to the gallery, for the offline tool
    std::vector<Observation> observeAll(const cv::Mat& image);

    // cosine similarity between two embeddings
    static float cosine(const cv::Mat& a, const cv::Mat& b);
    // how similar a feature is to the closest gallery face, -1 if the gallery is empty
    float similarity(const cv::Mat& feature) const;
    // the verdict for a face of this size and similarity
    static Verdict verdict(float similarity, float faceSize);

private:

    cv::Ptr<cv::FaceDetectorYN> detector;
    cv::Ptr<cv::FaceRecognizerSF> recognizer;
    std::vector<Reference> references;
    std::deque<cv::Mat> learned;

    // runs the detector on an image, rows of the result are faces (x, y, w, h, 5 landmarks, score)
    bool detect(const cv::Mat& image, cv::Mat& faces);
    cv::Mat feature(const cv::Mat& image, const cv::Mat& face);
    Observation describe(const cv::Mat& image, const cv::Mat& face, float scale, cv::Point2f offset);
};
}

#endif //FOLLOW_FACERECOGNIZER_H
