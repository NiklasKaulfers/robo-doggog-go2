#include "Control.h"

#include <cmath>

#define TOPIC_HIGHSTATE "rt/sportmodestate"
#define TOPIC_CONTROLLER "rt/wirelesscontroller"

// stick deflection below this is treated as noise
static constexpr float STICK_DEADZONE = 0.1f;
// how long the controller keeps priority after its last input
static constexpr auto CONTROLLER_HOLD_TIME = std::chrono::milliseconds(1500);

static const std::vector<std::string> FSM_STATE_STR = {
    "PASSIVE",
    "STAND_DOWN",
    "STAND_UP",
    "DEFAULT_MODE",
    "RUNNING_MODE",
    "CLIMB_MODE",
    "LEFT_SIDE_GAIT",
    "RIGHT_SIDE_GAIT",
    "HANDSTAND",
    "BIPED_STAND",
    "FRONT_FLIP",
    "BACK_FLIP",
    "RECOVERY",
    "BASE_HEIGHT_CTRL"
};

namespace robo::sdk::core
{
    void Control::init()
    {
        client.SetTimeout(2.0f);
        client.Init();
        client.WaitLeaseApplied();

        subState.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
        subState->InitChannel(std::bind(&Control::updateState, this, std::placeholders::_1), 1);

        subController.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_CONTROLLER));
        subController->InitChannel(std::bind(&Control::updateController, this, std::placeholders::_1), 1);
    }

    void Control::updateState(const void* message)
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        state = *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
    }

    void Control::updateController(const void* message)
    {
        const auto& input = *static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);
        const bool sticksMoved = std::abs(input.lx()) > STICK_DEADZONE || std::abs(input.ly()) > STICK_DEADZONE ||
                                 std::abs(input.rx()) > STICK_DEADZONE || std::abs(input.ry()) > STICK_DEADZONE;
        if (sticksMoved || input.keys() != 0)
            lastControllerInput = std::chrono::steady_clock::now().time_since_epoch().count();
    }

    bool Control::controllerActive() const
    {
        const std::chrono::steady_clock::time_point last {std::chrono::steady_clock::duration(lastControllerInput.load())};
        return lastControllerInput.load() != 0 && std::chrono::steady_clock::now() - last < CONTROLLER_HOLD_TIME;
    }

    void Control::printStatus()
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        std::cout << "Control status:" << std::endl;
        std::cout << "  position: " << state.position()[0] << ", " << state.position()[1] << ", " << state.position()[2] << std::endl;
        std::cout << "  temperature: " << static_cast<uint32_t>(state.imu_state().temperature()) << std::endl;
        std::cout << "  mode: " << FSM_STATE_STR[int(state.mode())] << std::endl;
        std::cout << "  progress: " << state.progress() << std::endl;
        std::cout << "  errorCode: " << state.error_code() << std::endl;
        std::cout << std::endl << std::flush;
    }

    void Control::hello()
    {
        client.Hello();
    }

    bool Control::standup()
    {
        bool ret = true;
        ret &= client.RecoveryStand();
        ret &= client.BalanceStand();
        ret &= client.FreeWalk();
        return ret;
    }

    bool Control::sitdown() {
        return client.Sit();
    }

    void Control::stopMovement() {
        client.StopMove();
    }

    void Control::rotateLeft(float speed)
    {
        client.Move(0.0, 0.0, speed);
    }

    void Control::rotateRight(float speed)
    {
        client.Move(0.0, 0.0, -speed);
    }

    void Control::moveForwardBackward(float speed)
    {
        client.Move(speed, 0.0, 0.0);
    }

    void Control::moveRightLeft(float speed)
    {
        client.Move(0.0, speed, 0.0);
    }

    void Control::moveRelative(float x, float z)
    {
        // one command so that translation and rotation are applied at the same time
        client.Move(x, 0.0f, z);
    }

    float Control::pitch() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        return state.imu_state().rpy()[1];
    }

    float Control::roll() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        return state.imu_state().rpy()[0];
    }

    float Control::positionZ() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        return state.position()[2];
    }
}
