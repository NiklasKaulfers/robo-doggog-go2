#ifndef FOLLOW_FOLLOWPERSON_H
#define FOLLOW_FOLLOWPERSON_H

#include "../core/Control.h"
#include "../core/ObstacleMap.h"
#include "../core/ObstacleMapSubscriber.h"
#include "Explore.h"
#include "unitree/robot/go2/video/video_client.hpp"
#include "vector"
#include <chrono>
#include <string>
#include "../follow/FollowPlanner.h"
#include "../ml/movenet/Multipose.h"
#include "../ml/PersonDetection.h"

namespace robo::sdk::examples {
// Follows one person. What to do is decided by the follow::FollowPlanner (who the target
// is, chasing them, searching when they are lost); this class connects it to the robot:
// camera and pose detector, remote controller priority, the explore combo, watchdogs,
// careful movement on stairs and debug frames.
class FollowPerson {

    core::Control& roboControl;
    unitree::robot::go2::VideoClient videoClient;
    std::vector<uint8_t> imgData;
    // whether we sent a movement command since the last stop
    bool moving = false;
    // on stairs only slow, straight, forward movement is sent
    bool stairMode = false;

    core::ObstacleMap obstacleMap;
    // feeds the LiDAR height map into obstacleMap
    core::ObstacleMapSubscriber mapSubscriber {obstacleMap, roboControl};

    // toggled with the explore button combo on the remote
    bool exploring = false;
    Explore explorer {roboControl, obstacleMap};

    // folder with photos of the person to follow, empty = follow whoever shows up first
    std::string targetImages;
    follow::FollowPlanner planner;

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
    // detects stairs from body tilt and height change, the gait itself is left to the robot
    void updateStairMode(int& stairSamples, float& levelHeight);
};
}

#endif //FOLLOW_FOLLOWPERSON_H
