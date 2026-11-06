#ifndef FOLLOW_CONTROL_H
#define FOLLOW_CONTROL_H

#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/idl/go2/WirelessController_.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"
#include "unitree/robot/go2/obstacles_avoid/obstacles_avoid_client.hpp"
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

    // all movement goes through the obstacle avoidance service. SportClient::Move
    // talks to the locomotion controller directly and skips avoidance entirely.
    unitree::robot::go2::ObstaclesAvoidClient avoidClient;
    // whether the avoidance service currently takes its commands from us instead of the remote
    bool apiControl = false;

public:

    // set by SIGINT/SIGTERM, tricks leave their loop so control is handed back cleanly
    static std::atomic<bool> stopRequested;

    unitree::robot::go2::SportClient client;

    ~Control();

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
    // stops and gives movement control back to the remote controller
    void releaseControl();

    // body orientation from the IMU in radians
    float pitch() const;
    float roll() const;
    // height of the body in the odometry frame, changes while climbing stairs
    float positionZ() const;
    // position and heading in the odometry frame
    float positionX() const;
    float positionY() const;
    float yaw() const;

    // true while someone is using the remote controller; tricks must not send
    // movement commands then, or they overwrite the controller's input
    bool controllerActive() const;

private:

    void updateState(const void *message);
    void updateController(const void *message);
    void move(float vx, float vy, float vyaw);
};
}

#endif //FOLLOW_CONTROL_H
