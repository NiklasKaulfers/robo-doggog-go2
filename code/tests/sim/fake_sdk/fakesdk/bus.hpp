#ifndef FAKESDK_BUS_HPP
#define FAKESDK_BUS_HPP

// The other side of the fake Unitree SDK and the LiteRT stub used by the simulation
// tests. Every call the robot code makes into the SDK is recorded here (with its
// arguments and when), results can be scripted (0 = success, like the real SDK), and the
// simulated world publishes DDS messages, camera images and pose detections through it.
// Process-wide, like the real SDK's DDS factory.

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <typeinfo>
#include <vector>

namespace fakesdk {

// one call into the fake SDK
struct Call {
    // seconds since the bus was reset
    double time = 0;
    // "<client>.<method>": avoid.*, sport.*, video.*, dds.*
    std::string api;
    std::vector<float> args;
    int32_t result = 0;
};

// what the obstacle avoidance service was told
struct Avoidance {
    // the avoidance itself (SwitchSet / SwitchGet)
    bool enabled = false;
    // takes movement commands from the API instead of the remote (UseRemoteCommandFromApi)
    bool apiControl = false;
    // the last velocity it accepted (Move while apiControl)
    float vx = 0, vy = 0, vyaw = 0;
};

class Bus {
public:
    static Bus& instance();

    // forgets calls, scripted results, the avoidance state and the sensor callbacks.
    // Subscriptions stay, they belong to objects that unsubscribe when destroyed
    void reset();

    // --- calls from the code under test ---

    // records a call, returns its scripted result (0 unless setResult was used)
    int32_t record(const std::string& api, std::vector<float> args = {});
    // result of every later call to api
    void setResult(const std::string& api, int32_t result);
    std::vector<Call> calls() const;
    std::vector<Call> calls(const std::string& api) const;
    size_t count(const std::string& api) const;
    // seconds since reset
    double now() const;

    // the avoidance service, updated by the fake ObstaclesAvoidClient
    Avoidance avoidance() const;
    void setAvoidanceEnabled(bool enabled);
    void setAvoidanceApiControl(bool apiControl);
    void setAvoidanceVelocity(float vx, float vy, float vyaw);

    // --- DDS ---

    using Handler = std::function<void(const void*)>;
    // returns an id for unsubscribe
    int subscribe(const std::string& topic, const std::type_info& type, Handler handler);
    // waits until no handler of this subscription runs anymore
    void unsubscribe(int id);
    bool hasSubscriber(const std::string& topic) const;
    // delivers to every subscriber of topic, on the calling thread (like a DDS thread)
    template <typename T>
    void publish(const std::string& topic, const T& message) { dispatch(topic, typeid(T), &message); }

    // --- sensors, provided by the simulated world ---

    // camera: fills a JPEG, returns 0 or an error code. Without one: error 1
    void setCamera(std::function<int32_t(std::vector<uint8_t>&)> camera);
    int32_t grabImage(std::vector<uint8_t>& out);
    // pose model inference: fills the MoveNet MultiPose output (6 x 56 floats). Without
    // one, or when it returns false, inference fails
    void setDetector(std::function<bool(std::vector<float>&)> detector);
    bool runDetector(std::vector<float>& output);
    // called when main() initializes DDS (ChannelFactory::Init)
    void setOnDdsInit(std::function<void()> callback);
    void ddsInit();

private:
    Bus() = default;
    void dispatch(const std::string& topic, const std::type_info& type, const void* message);

    struct Subscription {
        int id;
        std::string topic;
        const std::type_info* type;
        Handler handler;
    };

    mutable std::mutex mutex;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::vector<Call> log;
    std::map<std::string, int32_t> results;
    Avoidance avoid;
    std::function<int32_t(std::vector<uint8_t>&)> camera;
    std::function<bool(std::vector<float>&)> detector;
    std::function<void()> onDdsInit;

    // held while handlers run, so unsubscribe can wait for them
    mutable std::recursive_mutex dispatchMutex;
    std::vector<Subscription> subscriptions;
    int nextSubscription = 1;
};

inline Bus& bus() { return Bus::instance(); }
}

#endif //FAKESDK_BUS_HPP
