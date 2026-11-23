// The fake Unitree SDK: records every call on fakesdk::Bus and behaves like the real
// services where the robot code depends on it (obstacle avoidance state, DDS delivery)

#include <algorithm>
#include <stdexcept>
#include <fakesdk/bus.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/obstacles_avoid/obstacles_avoid_client.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/video/video_client.hpp>

namespace fakesdk {

Bus& Bus::instance() {
    static Bus bus;
    return bus;
}

void Bus::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    start = std::chrono::steady_clock::now();
    log.clear();
    results.clear();
    avoid = Avoidance();
    camera = nullptr;
    detector = nullptr;
    onDdsInit = nullptr;
}

double Bus::now() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int32_t Bus::record(const std::string& api, std::vector<float> args) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = results.find(api);
    const int32_t result = it == results.end() ? 0 : it->second;
    log.push_back(Call {now(), api, std::move(args), result});
    return result;
}

void Bus::setResult(const std::string& api, int32_t result) {
    std::lock_guard<std::mutex> lock(mutex);
    results[api] = result;
}

std::vector<Call> Bus::calls() const {
    std::lock_guard<std::mutex> lock(mutex);
    return log;
}

std::vector<Call> Bus::calls(const std::string& api) const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<Call> result;
    for (const auto& call : log)
        if (call.api == api) result.push_back(call);
    return result;
}

size_t Bus::count(const std::string& api) const {
    std::lock_guard<std::mutex> lock(mutex);
    return std::count_if(log.begin(), log.end(), [&](const Call& call) { return call.api == api; });
}

Avoidance Bus::avoidance() const {
    std::lock_guard<std::mutex> lock(mutex);
    return avoid;
}

void Bus::setAvoidanceEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex);
    avoid.enabled = enabled;
}

void Bus::setAvoidanceApiControl(bool apiControl) {
    std::lock_guard<std::mutex> lock(mutex);
    avoid.apiControl = apiControl;
    // back to the remote: the API's last command doesn't keep the robot moving
    if (!apiControl) avoid.vx = avoid.vy = avoid.vyaw = 0;
}

void Bus::setAvoidanceVelocity(float vx, float vy, float vyaw) {
    std::lock_guard<std::mutex> lock(mutex);
    // the service ignores API commands while it listens to the remote
    if (!avoid.apiControl) return;
    avoid.vx = vx;
    avoid.vy = vy;
    avoid.vyaw = vyaw;
}

int Bus::subscribe(const std::string& topic, const std::type_info& type, Handler handler) {
    std::lock_guard<std::recursive_mutex> dispatching(dispatchMutex);
    std::lock_guard<std::mutex> lock(mutex);
    const int id = nextSubscription++;
    subscriptions.push_back(Subscription {id, topic, &type, std::move(handler)});
    return id;
}

void Bus::unsubscribe(int id) {
    // waits for a dispatch in progress, the handler may belong to an object being destroyed
    std::lock_guard<std::recursive_mutex> dispatching(dispatchMutex);
    std::lock_guard<std::mutex> lock(mutex);
    subscriptions.erase(std::remove_if(subscriptions.begin(), subscriptions.end(),
                                       [&](const Subscription& s) { return s.id == id; }),
                        subscriptions.end());
}

bool Bus::hasSubscriber(const std::string& topic) const {
    std::lock_guard<std::mutex> lock(mutex);
    return std::any_of(subscriptions.begin(), subscriptions.end(), [&](const Subscription& s) { return s.topic == topic; });
}

void Bus::dispatch(const std::string& topic, const std::type_info& type, const void* message) {
    std::lock_guard<std::recursive_mutex> dispatching(dispatchMutex);
    std::vector<Handler> handlers;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto& s : subscriptions) {
            if (s.topic != topic) continue;
            if (*s.type != type) throw std::logic_error("message of the wrong type published on " + topic);
            handlers.push_back(s.handler);
        }
    }
    for (const auto& handler : handlers) handler(message);
}

void Bus::setCamera(std::function<int32_t(std::vector<uint8_t>&)> newCamera) {
    std::lock_guard<std::mutex> lock(mutex);
    camera = std::move(newCamera);
}

int32_t Bus::grabImage(std::vector<uint8_t>& out) {
    std::function<int32_t(std::vector<uint8_t>&)> current;
    {
        std::lock_guard<std::mutex> lock(mutex);
        current = camera;
    }
    if (!current) return 1;
    return current(out);
}

void Bus::setDetector(std::function<bool(std::vector<float>&)> newDetector) {
    std::lock_guard<std::mutex> lock(mutex);
    detector = std::move(newDetector);
}

bool Bus::runDetector(std::vector<float>& output) {
    std::function<bool(std::vector<float>&)> current;
    {
        std::lock_guard<std::mutex> lock(mutex);
        current = detector;
    }
    return current && current(output);
}

void Bus::setOnDdsInit(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(mutex);
    onDdsInit = std::move(callback);
}

void Bus::ddsInit() {
    std::function<void()> callback;
    {
        std::lock_guard<std::mutex> lock(mutex);
        callback = onDdsInit;
    }
    if (callback) callback();
}
}

using fakesdk::bus;

static float flag(bool value) { return value ? 1.0f : 0.0f; }

namespace unitree::robot {

ChannelFactory* ChannelFactory::Instance() {
    static ChannelFactory factory;
    return &factory;
}

void ChannelFactory::Init(int32_t domainId, const std::string&) {
    bus().record("dds.Init", {static_cast<float>(domainId)});
    bus().ddsInit();
}

void ChannelFactory::Init(const std::string&) {
    bus().record("dds.Init");
    bus().ddsInit();
}

void ChannelFactory::Release() {
    bus().record("dds.Release");
}

Client::Client(const std::string& name, bool) : name(name) {}
Client::~Client() = default;

void Client::SetTimeout(int64_t timeout) {
    bus().record(name + ".SetTimeout", {static_cast<float>(timeout)});
}

void Client::SetTimeout(float timeout) {
    bus().record(name + ".SetTimeout", {timeout});
}

void Client::WaitLeaseApplied() {
    bus().record(name + ".WaitLeaseApplied");
}
}

namespace unitree::robot::go2 {

// --- SportClient: everything recorded as sport.<Method> ---

SportClient::SportClient(bool) : Client("sport") {}
SportClient::~SportClient() = default;
void SportClient::Init() { bus().record("sport.Init"); }
int32_t SportClient::Damp() { return bus().record("sport.Damp"); }
int32_t SportClient::BalanceStand() { return bus().record("sport.BalanceStand"); }
int32_t SportClient::StopMove() { return bus().record("sport.StopMove"); }
int32_t SportClient::StandUp() { return bus().record("sport.StandUp"); }
int32_t SportClient::StandDown() { return bus().record("sport.StandDown"); }
int32_t SportClient::RecoveryStand() { return bus().record("sport.RecoveryStand"); }
int32_t SportClient::Euler(float roll, float pitch, float yaw) { return bus().record("sport.Euler", {roll, pitch, yaw}); }
int32_t SportClient::Move(float vx, float vy, float vyaw) { return bus().record("sport.Move", {vx, vy, vyaw}); }
int32_t SportClient::Sit() { return bus().record("sport.Sit"); }
int32_t SportClient::RiseSit() { return bus().record("sport.RiseSit"); }
int32_t SportClient::SpeedLevel(int level) { return bus().record("sport.SpeedLevel", {static_cast<float>(level)}); }
int32_t SportClient::Hello() { return bus().record("sport.Hello"); }
int32_t SportClient::Stretch() { return bus().record("sport.Stretch"); }
int32_t SportClient::SwitchJoystick(bool value) { return bus().record("sport.SwitchJoystick", {flag(value)}); }
int32_t SportClient::Content() { return bus().record("sport.Content"); }
int32_t SportClient::Heart() { return bus().record("sport.Heart"); }
int32_t SportClient::Pose(bool value) { return bus().record("sport.Pose", {flag(value)}); }
int32_t SportClient::Scrape() { return bus().record("sport.Scrape"); }
int32_t SportClient::FrontFlip() { return bus().record("sport.FrontFlip"); }
int32_t SportClient::FrontJump() { return bus().record("sport.FrontJump"); }
int32_t SportClient::FrontPounce() { return bus().record("sport.FrontPounce"); }
int32_t SportClient::Dance1() { return bus().record("sport.Dance1"); }
int32_t SportClient::Dance2() { return bus().record("sport.Dance2"); }
int32_t SportClient::LeftFlip() { return bus().record("sport.LeftFlip"); }
int32_t SportClient::BackFlip() { return bus().record("sport.BackFlip"); }
int32_t SportClient::HandStand(bool value) { return bus().record("sport.HandStand", {flag(value)}); }
int32_t SportClient::FreeWalk() { return bus().record("sport.FreeWalk"); }
int32_t SportClient::FreeBound(bool value) { return bus().record("sport.FreeBound", {flag(value)}); }
int32_t SportClient::FreeJump(bool value) { return bus().record("sport.FreeJump", {flag(value)}); }
int32_t SportClient::FreeAvoid(bool value) { return bus().record("sport.FreeAvoid", {flag(value)}); }
int32_t SportClient::ClassicWalk(bool value) { return bus().record("sport.ClassicWalk", {flag(value)}); }
int32_t SportClient::WalkUpright(bool value) { return bus().record("sport.WalkUpright", {flag(value)}); }
int32_t SportClient::CrossStep(bool value) { return bus().record("sport.CrossStep", {flag(value)}); }
int32_t SportClient::AutoRecoverSet(bool value) { return bus().record("sport.AutoRecoverSet", {flag(value)}); }
int32_t SportClient::AutoRecoverGet(bool& value) {
    value = true;
    return bus().record("sport.AutoRecoverGet");
}
int32_t SportClient::StaticWalk() { return bus().record("sport.StaticWalk"); }
int32_t SportClient::TrotRun() { return bus().record("sport.TrotRun"); }
int32_t SportClient::EconomicGait() { return bus().record("sport.EconomicGait"); }
int32_t SportClient::SwitchAvoidMode() { return bus().record("sport.SwitchAvoidMode"); }

// --- ObstaclesAvoidClient: recorded as avoid.<Method>, keeps the service state ---

ObstaclesAvoidClient::ObstaclesAvoidClient() : Client("avoid") {}
ObstaclesAvoidClient::~ObstaclesAvoidClient() = default;
void ObstaclesAvoidClient::Init() { bus().record("avoid.Init"); }

int32_t ObstaclesAvoidClient::SwitchSet(bool enable) {
    const int32_t result = bus().record("avoid.SwitchSet", {flag(enable)});
    if (result == 0) bus().setAvoidanceEnabled(enable);
    return result;
}

int32_t ObstaclesAvoidClient::SwitchGet(bool& enable) {
    const int32_t result = bus().record("avoid.SwitchGet");
    if (result == 0) enable = bus().avoidance().enabled;
    return result;
}

int32_t ObstaclesAvoidClient::Move(float x, float y, float yaw) {
    const int32_t result = bus().record("avoid.Move", {x, y, yaw});
    if (result == 0) bus().setAvoidanceVelocity(x, y, yaw);
    return result;
}

int32_t ObstaclesAvoidClient::UseRemoteCommandFromApi(bool isRemoteCommandsFromApi) {
    const int32_t result = bus().record("avoid.UseRemoteCommandFromApi", {flag(isRemoteCommandsFromApi)});
    if (result == 0) bus().setAvoidanceApiControl(isRemoteCommandsFromApi);
    return result;
}

int32_t ObstaclesAvoidClient::MoveToAbsolutePosition(float x, float y, float yaw) {
    return bus().record("avoid.MoveToAbsolutePosition", {x, y, yaw});
}

int32_t ObstaclesAvoidClient::MoveToIncrementPosition(float x, float y, float yaw) {
    return bus().record("avoid.MoveToIncrementPosition", {x, y, yaw});
}

// --- VideoClient: images come from the simulated world ---

VideoClient::VideoClient() : Client("video") {}
VideoClient::~VideoClient() = default;
void VideoClient::Init() { bus().record("video.Init"); }

int32_t VideoClient::GetImageSample(std::vector<uint8_t>& image) {
    const int32_t result = bus().record("video.GetImageSample");
    if (result != 0) return result;
    return bus().grabImage(image);
}
}
