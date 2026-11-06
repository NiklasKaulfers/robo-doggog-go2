#ifndef FOLLOW_FOLLOWPERSON_H
#define FOLLOW_FOLLOWPERSON_H

#include "../core/Control.h"
#include "unitree/robot/go2/video/video_client.hpp"
#include "vector"
#include "../ml/movenet/Multipose.h"

namespace robo::sdk::examples {
class FollowPerson {

    core::Control& roboControl;
    unitree::robot::go2::VideoClient videoClient;
    std::vector<uint8_t> imgData;
    // whether we sent a movement command since the last stop
    bool moving = false;
    // on stairs only slow, straight, forward movement is sent
    bool stairMode = false;

public:

    explicit FollowPerson(core::Control& control);

    void run();

private:

    void runWithMultipose();
    void stop();
    // detects stairs from body tilt and height change, the gait itself is left to the robot
    void updateStairMode(int& stairSamples, float& levelHeight);
};
}

#endif //FOLLOW_FOLLOWPERSON_H
