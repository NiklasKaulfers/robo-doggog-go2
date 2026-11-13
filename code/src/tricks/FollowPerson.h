#ifndef FOLLOW_FOLLOWPERSON_H
#define FOLLOW_FOLLOWPERSON_H

#include "../core/Control.h"
#include "../core/ObstacleMap.h"
#include "Explore.h"
#include "unitree/robot/go2/video/video_client.hpp"
#include "vector"
#include <chrono>
#include <string>
#include "../ml/movenet/Multipose.h"
#include "../ml/PersonDetection.h"
#include "../tracking/TargetTracker.h"

namespace robo::sdk::examples {
// Follows one person. Who that is and where they are is the TargetTracker's job (face
// recognition against reference photos, clothes learned from the camera), this class
// does the walking: keeping the distance, going around obstacles, finding them again
// where they went, and being careful on stairs.
class FollowPerson {

    core::Control& roboControl;
    unitree::robot::go2::VideoClient videoClient;
    std::vector<uint8_t> imgData;
    // whether we sent a movement command since the last stop
    bool moving = false;
    // on stairs only slow, straight, forward movement is sent
    bool stairMode = false;

    core::ObstacleMap obstacleMap;
    // 1 = left, -1 = right; detours keep going around the same side
    int lastDetourSide = 1;

    // toggled with the explore button combo on the remote
    bool exploring = false;
    Explore explorer {roboControl, obstacleMap};

    // roaming around to find the lost person after looking where they went didn't help
    bool roamSearching = false;

    // folder with photos of the person to follow, empty = follow whoever shows up first
    std::string targetImages;
    tracking::TargetTracker tracker;

    // folder where annotated camera frames are written for debugging, empty = off
    std::string debugDir;
    std::chrono::steady_clock::time_point lastDebugFrame;
    int debugFrameCount = 0;

public:

    // with targetImages only the person shown in the photos in that folder is followed
    explicit FollowPerson(core::Control& control, std::string targetImages = "", std::string debugDir = "");

    void run();

private:

    void runWithMultipose();
    void stop();
    // sends a movement, limited to what is safe on stairs
    void drive(float x, float z);
    // everyone the pose detector found, in frame pixels
    static void collectDetections(const ml::movenet::Multipose& detector, std::vector<ml::PersonDetection>& out);
    // writes an annotated frame to debugDir now and then
    void saveDebugFrame(const cv::Mat& frame, std::chrono::steady_clock::time_point now);
    // heading (relative to the robot) toward the person that avoids obstacles in the
    // remembered map; clearance is set to how far the robot can walk along it
    float planHeading(float personAngle, float distance, float& clearance);
    // detects stairs from body tilt and height change, the gait itself is left to the robot
    void updateStairMode(int& stairSamples, float& levelHeight);
};
}

#endif //FOLLOW_FOLLOWPERSON_H
