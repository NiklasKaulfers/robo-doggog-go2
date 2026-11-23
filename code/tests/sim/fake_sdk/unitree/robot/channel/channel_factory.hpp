#ifndef FAKESDK_CHANNEL_FACTORY_HPP
#define FAKESDK_CHANNEL_FACTORY_HPP

// fake of the Unitree SDK's DDS factory: Init is recorded as "dds.Init" and lets the
// simulated world start (fakesdk::Bus::setOnDdsInit)

#include <cstdint>
#include <string>

namespace unitree::robot {
class ChannelFactory {
public:
    static ChannelFactory* Instance();

    void Init(int32_t domainId, const std::string& networkInterface = "");
    void Init(const std::string& configFileName = "");
    void Release();

private:
    ChannelFactory() = default;
};
}

#endif //FAKESDK_CHANNEL_FACTORY_HPP
