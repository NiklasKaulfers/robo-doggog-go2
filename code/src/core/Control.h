#ifndef FOLLOW_CONTROL_H
#define FOLLOW_CONTROL_H

#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace robo::sdk::core
{
class Control
{

    unitree_go::msg::dds_::SportModeState_ state;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> subState;

public:

    unitree::robot::go2::SportClient client;

    void init();
    void printStatus();
    void hello();
    bool standup();
    bool sitdown();
    void stopMovement();
    void rotateLeft(float speed);
    void rotateRight(float speed);
    void moveForwardBackward(float speed);
    void moveRightLeft(float speed);

private:

    void updateState(const void *message);
};
}

#endif //FOLLOW_CONTROL_H
