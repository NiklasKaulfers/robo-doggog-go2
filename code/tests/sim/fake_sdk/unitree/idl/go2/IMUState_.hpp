#ifndef FAKESDK_IMUSTATE_HPP
#define FAKESDK_IMUSTATE_HPP

// fake of the Unitree SDK's IDL type, same accessors as the generated one

#include <array>
#include <cstdint>

namespace unitree_go::msg::dds_ {
class IMUState_ {
    std::array<float, 4> quaternion_ {};
    std::array<float, 3> gyroscope_ {};
    std::array<float, 3> accelerometer_ {};
    std::array<float, 3> rpy_ {};
    uint8_t temperature_ = 0;

public:
    const std::array<float, 4>& quaternion() const { return quaternion_; }
    std::array<float, 4>& quaternion() { return quaternion_; }
    void quaternion(const std::array<float, 4>& value) { quaternion_ = value; }
    const std::array<float, 3>& gyroscope() const { return gyroscope_; }
    std::array<float, 3>& gyroscope() { return gyroscope_; }
    void gyroscope(const std::array<float, 3>& value) { gyroscope_ = value; }
    const std::array<float, 3>& accelerometer() const { return accelerometer_; }
    std::array<float, 3>& accelerometer() { return accelerometer_; }
    void accelerometer(const std::array<float, 3>& value) { accelerometer_ = value; }
    // roll, pitch, yaw (rad)
    const std::array<float, 3>& rpy() const { return rpy_; }
    std::array<float, 3>& rpy() { return rpy_; }
    void rpy(const std::array<float, 3>& value) { rpy_ = value; }
    uint8_t temperature() const { return temperature_; }
    uint8_t& temperature() { return temperature_; }
    void temperature(uint8_t value) { temperature_ = value; }
};
}

#endif //FAKESDK_IMUSTATE_HPP
