#ifndef FOLLOW_TURNTOPERSON_H
#define FOLLOW_TURNTOPERSON_H

#include "../core/Control.h"
#include "unitree/robot/go2/video/video_client.hpp"
#include "vector"

namespace robo::sdk::examples {
class TurnToPerson {

    core::Control& roboControl;
    unitree::robot::go2::VideoClient videoClient;
    std::vector<uint8_t> imgData;
    int countNotFound = 0;

public:

    explicit TurnToPerson(core::Control& control);

    void run();

private:

    void runWithSinglepose();
    void runWithMultipose();
};
}

#endif //FOLLOW_TURNTOPERSON_H
