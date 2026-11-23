#ifndef FAKESDK_HEIGHTMAP_HPP
#define FAKESDK_HEIGHTMAP_HPP

// fake of the Unitree SDK's IDL type, same accessors as the generated one

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace unitree_go::msg::dds_ {
class HeightMap_ {
    double stamp_ = 0;
    std::string frame_id_;
    float resolution_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::array<float, 2> origin_ {};
    std::vector<float> data_;

public:
    double stamp() const { return stamp_; }
    double& stamp() { return stamp_; }
    void stamp(double value) { stamp_ = value; }
    const std::string& frame_id() const { return frame_id_; }
    std::string& frame_id() { return frame_id_; }
    void frame_id(const std::string& value) { frame_id_ = value; }
    float resolution() const { return resolution_; }
    float& resolution() { return resolution_; }
    void resolution(float value) { resolution_ = value; }
    uint32_t width() const { return width_; }
    uint32_t& width() { return width_; }
    void width(uint32_t value) { width_ = value; }
    uint32_t height() const { return height_; }
    uint32_t& height() { return height_; }
    void height(uint32_t value) { height_ = value; }
    // world position of the corner of cell (0, 0)
    const std::array<float, 2>& origin() const { return origin_; }
    std::array<float, 2>& origin() { return origin_; }
    void origin(const std::array<float, 2>& value) { origin_ = value; }
    // row-major heights, unobserved cells are huge
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }
    void data(const std::vector<float>& value) { data_ = value; }
};
}

#endif //FAKESDK_HEIGHTMAP_HPP
