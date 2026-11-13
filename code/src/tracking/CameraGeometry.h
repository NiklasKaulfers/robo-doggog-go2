#ifndef FOLLOW_CAMERAGEOMETRY_H
#define FOLLOW_CAMERAGEOMETRY_H

#include <cmath>
#include "opencv2/core.hpp"

namespace robo::sdk::tracking {

// Where a person is, from where they appear in the Go2's front camera.
struct CameraGeometry {
    // horizontal field of view of the front camera, maps image position to an angle
    static constexpr float FOV = 2.09f;             // 120 degrees

    // The width of a standing person as a fraction of the frame width when they stand
    // about 0.5m in front of the camera. Boxes scale roughly inversely with the distance
    // to the camera, so comparing the detected box width against this gives an estimate
    // of the distance. Calibrated for a ~0.4m wide person (shoulders) in front of the
    // Go2's 120° camera, tune per installation.
    static constexpr float CALIBRATION_DISTANCE = 0.5f;
    static constexpr float BOX_WIDTH_AT_CALIBRATION = 0.35f;

    // direction of the box relative to the robot's heading, positive = left
    static float angle(const cv::Rect2f& box, int frameWidth) {
        const float center = (box.x + box.width / 2) / frameWidth;
        return -(center - 0.5f) * FOV;
    }

    // estimated distance (m) to the person in the box. The apparent width scales roughly
    // inversely with the distance, which is a more sensitive measure than the height
    static float distance(const cv::Rect2f& box, int frameWidth) {
        const float width = std::max(box.width / frameWidth, 1e-3f);
        return CALIBRATION_DISTANCE * BOX_WIDTH_AT_CALIBRATION / width;
    }

    // horizontal pixel shift of everything in view when the robot turns by yawChange
    static float pixelShift(float yawChange, int frameWidth) {
        return -yawChange / FOV * frameWidth;
    }

    static float wrapAngle(float a) {
        return std::atan2(std::sin(a), std::cos(a));
    }
};
}

#endif //FOLLOW_CAMERAGEOMETRY_H
