#ifndef FOLLOW_PERSONDETECTION_H
#define FOLLOW_PERSONDETECTION_H

#include <array>
#include "opencv2/core.hpp"

namespace robo::sdk::ml {

// A person found by the pose detector, in pixels of the full camera frame. This is the
// only thing the recognition and tracking code knows about the detector, so it can be
// fed from any pose model (and from still images in the offline tools).
struct PersonDetection {
    static constexpr int KEYPOINT_COUNT = 17;

    // COCO keypoint order, as used by Movenet
    enum Keypoint {
        NOSE = 0,
        LEFT_EYE, RIGHT_EYE,
        LEFT_EAR, RIGHT_EAR,
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    };

    cv::Rect2f box;
    float score = 0;
    std::array<cv::Point2f, KEYPOINT_COUNT> keypoints {};
    std::array<float, KEYPOINT_COUNT> confidence {};

    bool has(int keypoint, float minConfidence) const {
        return confidence[keypoint] >= minConfidence;
    }

    cv::Point2f center() const {
        return {box.x + box.width / 2, box.y + box.height / 2};
    }
};
}

#endif //FOLLOW_PERSONDETECTION_H
