#ifndef FAKESDK_WIRELESSCONTROLLER_HPP
#define FAKESDK_WIRELESSCONTROLLER_HPP

// fake of the Unitree SDK's IDL type, same accessors as the generated one

#include <cstdint>

namespace unitree_go::msg::dds_ {
class WirelessController_ {
    float lx_ = 0, ly_ = 0, rx_ = 0, ry_ = 0;
    uint16_t keys_ = 0;

public:
    float lx() const { return lx_; }
    float& lx() { return lx_; }
    void lx(float value) { lx_ = value; }
    float ly() const { return ly_; }
    float& ly() { return ly_; }
    void ly(float value) { ly_ = value; }
    float rx() const { return rx_; }
    float& rx() { return rx_; }
    void rx(float value) { rx_ = value; }
    float ry() const { return ry_; }
    float& ry() { return ry_; }
    void ry(float value) { ry_ = value; }
    // button bits, see unitree_joystick.hpp (R1 = bit 0, L1 = bit 1, Y = bit 11, ...)
    uint16_t keys() const { return keys_; }
    uint16_t& keys() { return keys_; }
    void keys(uint16_t value) { keys_ = value; }
};
}

#endif //FAKESDK_WIRELESSCONTROLLER_HPP
