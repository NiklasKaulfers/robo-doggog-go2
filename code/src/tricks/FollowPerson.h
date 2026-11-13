#ifndef FOLLOW_FOLLOWPERSON_H
#define FOLLOW_FOLLOWPERSON_H

#include "../core/Control.h"
#include "../core/ObstacleMap.h"
#include "Explore.h"
#include "unitree/robot/go2/video/video_client.hpp"
#include "vector"
#include <chrono>
#include "../ml/movenet/Multipose.h"
#include "../ml/PersonIdentifier.h"
#include <string>

namespace robo::sdk::examples {
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

    // last known position (odometry frame) and velocity of the person we follow
    float personX = 0, personY = 0;
    float personVX = 0, personVY = 0;
    std::chrono::steady_clock::time_point lastSeen;
    // 1 = left, -1 = right; where to look first when the person is gone
    int lostSide = 1;
    // roaming around to find the lost person after looking where they went didn't help
    bool roamSearching = false;

    // folder with images of the person to follow, empty = follow whoever shows up first
    std::string targetImages;
    ml::PersonIdentifier identifier;
    bool identify = false;

public:

    // with targetImages only the person shown in the images in that folder is followed
    explicit FollowPerson(core::Control& control, std::string targetImages = "");

    void run();

private:

    void runWithMultipose();
    void stop();
    // sends a movement, limited to what is safe on stairs
    void drive(float x, float z);
    // remembers where the person is and how fast they walk, to predict them when lost
    void updatePersonTrack(float x, float y, bool continuous, std::chrono::steady_clock::time_point now);
    // heading (relative to the robot) toward the person that avoids obstacles in the
    // remembered map; clearance is set to how far the robot can walk along it
    float planHeading(float personAngle, float distance, float& clearance);
    // detects stairs from body tilt and height change, the gait itself is left to the robot
    void updateStairMode(int& stairSamples, float& levelHeight);
};
}

#endif //FOLLOW_FOLLOWPERSON_H
