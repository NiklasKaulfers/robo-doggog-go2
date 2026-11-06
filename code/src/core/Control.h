#ifndef FOLLOW_CONTROL_H
#define FOLLOW_CONTROL_H

#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/idl/go2/WirelessController_.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <atomic>
#include <chrono>
#include <mutex>

namespace robo::sdk::core
{
class Control
{

    unitree_go::msg::dds_::SportModeState_ state;
    // state is written by the DDS thread and read by the trick loop
    mutable std::mutex stateMutex;

    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> subController;
    // last time a stick was moved or a button pressed on the remote, in steady_clock ticks
    std::atomic<std::chrono::steady_clock::rep> lastControllerInput {0};
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

    // true while someone is using the remote controller; tricks must not send
    // movement commands then, or they overwrite the controller's input
    bool controllerActive() const;

private:

    void updateState(const void *message);
    void updateController(const void *message);
};
}

#endif //FOLLOW_CONTROL_H
