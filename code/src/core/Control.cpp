#include "Control.h"

#include <cmath>
#include <string>
#include <vector>
#include "../logging/Logger.h"

#define TOPIC_HIGHSTATE "rt/sportmodestate"
#define TOPIC_CONTROLLER "rt/wirelesscontroller"

namespace logging = robo::sdk::logging;

// stick deflection below this is treated as noise
static constexpr float STICK_DEADZONE = 0.1f;
// how long the controller keeps priority after its last input
static constexpr auto CONTROLLER_HOLD_TIME = std::chrono::milliseconds(1500);

// button bits of WirelessController_::keys(), see unitree_joystick.hpp
static constexpr uint16_t KEY_R1 = 1 << 0;
static constexpr uint16_t KEY_L1 = 1 << 1;
static constexpr uint16_t KEY_Y = 1 << 11;
// toggles exploring; must not be a combo the Go2 firmware already uses
static constexpr uint16_t EXPLORE_COMBO = KEY_L1 | KEY_R1 | KEY_Y;

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
    std::atomic<bool> Control::stopRequested {false};

    void Control::init()
    {
        client.SetTimeout(2.0f);
        client.Init();
        client.WaitLeaseApplied();

        subState.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
        subState->InitChannel(std::bind(&Control::updateState, this, std::placeholders::_1), 1);

        subController.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_CONTROLLER));
        subController->InitChannel(std::bind(&Control::updateController, this, std::placeholders::_1), 1);

        avoidClient.SetTimeout(2.0f);
        avoidClient.Init();
        // avoidance is only ever switched on here, never off
        bool avoidEnabled = false;
        if (avoidClient.SwitchGet(avoidEnabled) != 0 || !avoidEnabled) {
            if (avoidClient.SwitchSet(true) != 0)
                logging::error("Control", "could not enable obstacle avoidance");
        }
    }

    Control::~Control()
    {
        releaseControl();
    }

    void Control::updateState(const void* message)
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        state = *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
        lastStateTime = std::chrono::steady_clock::now().time_since_epoch().count();
    }

    float Control::stateAge() const
    {
        const auto last = lastStateTime.load();
        if (last == 0) return INFINITY;
        const std::chrono::steady_clock::time_point time {std::chrono::steady_clock::duration(last)};
        return std::chrono::duration<float>(std::chrono::steady_clock::now() - time).count();
    }

    void Control::updateController(const void* message)
    {
        const auto& input = *static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);
        const bool sticksMoved = std::abs(input.lx()) > STICK_DEADZONE || std::abs(input.ly()) > STICK_DEADZONE ||
                                 std::abs(input.rx()) > STICK_DEADZONE || std::abs(input.ry()) > STICK_DEADZONE;
        if (sticksMoved || input.keys() != 0)
            lastControllerInput = std::chrono::steady_clock::now().time_since_epoch().count();

        // count a press only when the combo is newly complete, not while it's held
        const bool comboHeld = (input.keys() & EXPLORE_COMBO) == EXPLORE_COMBO;
        if (comboHeld && !exploreComboHeld) explorePresses++;
        exploreComboHeld = comboHeld;
    }

    bool Control::controllerActive() const
    {
        const std::chrono::steady_clock::time_point last {std::chrono::steady_clock::duration(lastControllerInput.load())};
        return lastControllerInput.load() != 0 && std::chrono::steady_clock::now() - last < CONTROLLER_HOLD_TIME;
    }

    bool Control::exploreComboPressed()
    {
        int presses = explorePresses.load();
        while (presses > 0) {
            if (explorePresses.compare_exchange_weak(presses, presses - 1)) return true;
        }
        return false;
    }

    void Control::printStatus()
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        const std::string position = logging::text(state.position()[0], ", ", state.position()[1], ", ", state.position()[2]);
        const std::string temperature = std::to_string(static_cast<uint32_t>(state.imu_state().temperature()));
        // the firmware may report modes newer than this table
        const int modeNumber = int(state.mode());
        const std::string mode = modeNumber >= 0 && modeNumber < int(FSM_STATE_STR.size())
                                     ? FSM_STATE_STR[modeNumber] : "UNKNOWN (" + std::to_string(modeNumber) + ")";
        const std::string progress = logging::text(state.progress());
        const std::string errorCode = logging::text(state.error_code());
        logging::info("Control", "status: position " + position + ", temperature " + temperature + ", mode " + mode
                                     + ", progress " + progress + ", errorCode " + errorCode,
                      {{"position", position}, {"temperature", temperature}, {"mode", mode},
                       {"progress", progress}, {"errorCode", errorCode}});
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

    bool Control::stopMovement()
    {
        if (!apiControl) return true;
        if (avoidClient.Move(0.0f, 0.0f, 0.0f) == 0) {
            stopFailureLogged = false;
            return true;
        }
        if (!stopFailureLogged) {
            logging::warn("Control", "stop command failed, retrying");
            stopFailureLogged = true;
        }
        return false;
    }

    void Control::rotateLeft(float speed)
    {
        move(0.0f, 0.0f, speed);
    }

    void Control::rotateRight(float speed)
    {
        move(0.0f, 0.0f, -speed);
    }

    void Control::moveForwardBackward(float speed)
    {
        move(speed, 0.0f, 0.0f);
    }

    void Control::moveRightLeft(float speed)
    {
        move(0.0f, speed, 0.0f);
    }

    void Control::moveRelative(float x, float z)
    {
        // one command so that translation and rotation are applied at the same time
        move(x, 0.0f, z);
    }

    void Control::move(float vx, float vy, float vyaw)
    {
        if (!apiControl) {
            if (avoidClient.UseRemoteCommandFromApi(true) != 0) {
                // never fall back to SportClient::Move, that would bypass avoidance
                logging::error("Control", "avoidance service unavailable, not moving");
                return;
            }
            apiControl = true;
        }
        if (avoidClient.Move(vx, vy, vyaw) != 0)
            logging::error("Control", "error in avoidance move");
    }

    void Control::releaseControl()
    {
        if (!apiControl) return;
        avoidClient.Move(0.0f, 0.0f, 0.0f);
        avoidClient.UseRemoteCommandFromApi(false);
        apiControl = false;
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

    float Control::positionX() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        return state.position()[0];
    }

    float Control::positionY() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        return state.position()[1];
    }

    float Control::yaw() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        return state.imu_state().rpy()[2];
    }

    float Control::positionZ() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        return state.position()[2];
    }
}
