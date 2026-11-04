#ifndef FOLLOW_SITSTANDAPPROACH_H
#define FOLLOW_SITSTANDAPPROACH_H

#include "../core/Control.h"
#include "unitree/robot/go2/video/video_client.hpp"
#include "vector"
#include "../ml/movenet/Movenet.h"

namespace robo::sdk::examples {
class SitStandApproach {

    core::Control& roboControl;
    unitree::robot::go2::VideoClient videoClient;
    std::vector<uint8_t> imgData;

public:

    explicit SitStandApproach(core::Control& control);

    void run();

private:

    void runWithSinglepose();

    bool checkGestureSit(const ml::movenet::Movenet& detector);
    bool checkGestureStand(const ml::movenet::Movenet& detector);
    bool checkGestureApproach(const ml::movenet::Movenet& detector);
};
}

#endif //FOLLOW_SITSTANDAPPROACH_H
