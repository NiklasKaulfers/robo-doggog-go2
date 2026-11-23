#ifndef FAKESDK_SPORTMODESTATE_HPP
#define FAKESDK_SPORTMODESTATE_HPP

// fake of the Unitree SDK's IDL type, same accessors as the generated one (only the
// fields the app reads, plus a few neighbours)

#include <array>
#include <cstdint>
#include "IMUState_.hpp"

namespace unitree_go::msg::dds_ {
class SportModeState_ {
    uint32_t error_code_ = 0;
    IMUState_ imu_state_;
    uint8_t mode_ = 0;
    float progress_ = 0;
    uint8_t gait_type_ = 0;
    std::array<float, 3> position_ {};
    float body_height_ = 0;
    std::array<float, 3> velocity_ {};
    float yaw_speed_ = 0;

public:
    uint32_t error_code() const { return error_code_; }
    uint32_t& error_code() { return error_code_; }
    void error_code(uint32_t value) { error_code_ = value; }
    const IMUState_& imu_state() const { return imu_state_; }
    IMUState_& imu_state() { return imu_state_; }
    void imu_state(const IMUState_& value) { imu_state_ = value; }
    uint8_t mode() const { return mode_; }
    uint8_t& mode() { return mode_; }
    void mode(uint8_t value) { mode_ = value; }
    float progress() const { return progress_; }
    float& progress() { return progress_; }
    void progress(float value) { progress_ = value; }
    uint8_t gait_type() const { return gait_type_; }
    uint8_t& gait_type() { return gait_type_; }
    void gait_type(uint8_t value) { gait_type_ = value; }
    // odometry position (m)
    const std::array<float, 3>& position() const { return position_; }
    std::array<float, 3>& position() { return position_; }
    void position(const std::array<float, 3>& value) { position_ = value; }
    float body_height() const { return body_height_; }
    float& body_height() { return body_height_; }
    void body_height(float value) { body_height_ = value; }
    const std::array<float, 3>& velocity() const { return velocity_; }
    std::array<float, 3>& velocity() { return velocity_; }
    void velocity(const std::array<float, 3>& value) { velocity_ = value; }
    float yaw_speed() const { return yaw_speed_; }
    float& yaw_speed() { return yaw_speed_; }
    void yaw_speed(float value) { yaw_speed_ = value; }
};
}

#endif //FAKESDK_SPORTMODESTATE_HPP
