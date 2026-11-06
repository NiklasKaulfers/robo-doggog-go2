#ifndef FOLLOW_CONTROL_H
#define FOLLOW_CONTROL_H

#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <mutex>

namespace robo::sdk::core
{
class Control
{

    unitree_go::msg::dds_::SportModeState_ state;
    // state is written by the DDS thread and read by the trick loop
    mutable std::mutex stateMutex;
    bool stairMode = false;
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
    void moveRelative(float x, float z);

    // body orientation from the IMU in radians
    float pitch() const;
    float roll() const;
    // height of the body in the odometry frame, changes while climbing stairs
    float positionZ() const;

    // slow, careful gait for stairs; normal gait otherwise
    bool setStairMode(bool enabled);
    bool isStairMode() const { return stairMode; }

private:

    void updateState(const void *message);
};
}

#endif //FOLLOW_CONTROL_H
