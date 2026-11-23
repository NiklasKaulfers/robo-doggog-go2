#ifndef FAKESDK_CHANNEL_SUBSCRIBER_HPP
#define FAKESDK_CHANNEL_SUBSCRIBER_HPP

// fake of the Unitree SDK's DDS subscriber: subscribes on fakesdk::Bus, where the
// simulated world publishes. Unsubscribes when destroyed (the real one closes its channel)

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <fakesdk/bus.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

namespace unitree::robot {
template <typename MSG>
class ChannelSubscriber {
public:
    explicit ChannelSubscriber(const std::string& channelName) : channelName(channelName) {}
    explicit ChannelSubscriber(const std::string& channelName, const std::function<void(const void*)>& handler,
                               int64_t queuelen = 0)
        : channelName(channelName), handler(handler) {}
    ~ChannelSubscriber() { CloseChannel(); }
    ChannelSubscriber(const ChannelSubscriber&) = delete;
    ChannelSubscriber& operator=(const ChannelSubscriber&) = delete;

    void InitChannel(const std::function<void(const void*)>& newHandler, int64_t queuelen = 0) {
        handler = newHandler;
        InitChannel();
    }
    void InitChannel() {
        if (!handler) throw std::invalid_argument("subscribe handler is invalid");
        CloseChannel();
        subscription = fakesdk::bus().subscribe(channelName, typeid(MSG), handler);
    }
    void CloseChannel() {
        if (subscription) fakesdk::bus().unsubscribe(subscription);
        subscription = 0;
    }
    const std::string& GetChannelName() const { return channelName; }

private:
    std::string channelName;
    std::function<void(const void*)> handler;
    int subscription = 0;
};

template <typename MSG>
using ChannelSubscriberPtr = std::shared_ptr<ChannelSubscriber<MSG>>;
}

#endif //FAKESDK_CHANNEL_SUBSCRIBER_HPP
