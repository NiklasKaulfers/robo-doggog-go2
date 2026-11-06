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

public:

    explicit FollowPerson(core::Control& control);

    void run();

private:

    void runWithMultipose();
    // switches the robot into/out of stair mode based on body tilt and height change
    void updateStairMode(int& stairSamples, float& levelHeight);
};
}

#endif //FOLLOW_FOLLOWPERSON_H
