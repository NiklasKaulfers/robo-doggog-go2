#ifndef FAKESDK_VIDEO_CLIENT_HPP
#define FAKESDK_VIDEO_CLIENT_HPP

// fake of the Unitree SDK's front camera client, images come from fakesdk::Bus

#include <cstdint>
#include <vector>
#include <unitree/robot/client/client.hpp>

namespace unitree::robot::go2 {
class VideoClient : public Client {
public:
    explicit VideoClient();
    ~VideoClient() override;

    void Init() override;
    // a JPEG of the front camera; 0 = success
    int32_t GetImageSample(std::vector<uint8_t>& image);
};
}

#endif //FAKESDK_VIDEO_CLIENT_HPP
