#ifndef FOLLOW_TESTS_SIMRUN_H
#define FOLLOW_TESTS_SIMRUN_H

// One run of the follow app against the simulated world, like main() after the robot
// stood up: Control connects to the (fake) robot and FollowPerson runs its loop on a
// thread of its own, in real time. The commands it sends are recorded on fakesdk::Bus.

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <fakesdk/bus.hpp>
#include "SimWorld.h"
#include "core/Control.h"
#include "tricks/FollowPerson.h"

namespace sim {

// a movement command sent to the obstacle avoidance service
struct Move {
    double time;
    float x, y, z;
    bool stop() const { return x == 0 && y == 0 && z == 0; }
};

// the movement commands sent between from and to (bus clock)
inline std::vector<Move> moves(double from = 0, double to = 1e9) {
    std::vector<Move> result;
    for (const auto& call : fakesdk::bus().calls("avoid.Move")) {
        if (call.time < from || call.time > to || call.args.size() != 3) continue;
        result.push_back({call.time, call.args[0], call.args[1], call.args[2]});
    }
    return result;
}

// calls to one api between from and to (bus clock)
inline std::vector<fakesdk::Call> callsTo(const std::string& api, double from = 0, double to = 1e9) {
    std::vector<fakesdk::Call> result;
    for (const auto& call : fakesdk::bus().calls(api))
        if (call.time >= from && call.time <= to) result.push_back(call);
    return result;
}

class SimRun {
public:
    SimWorld world;

    SimRun() {
        fakesdk::bus().reset();
        robo::sdk::core::Control::stopRequested = false;
    }
    ~SimRun() { shutdown(); }
    SimRun(const SimRun&) = delete;
    SimRun& operator=(const SimRun&) = delete;

    // starts the world, connects to the robot and runs the follow loop
    void start(const std::string& targetImages = "", const std::string& debugDir = "") {
        world.start();
        control = std::make_unique<robo::sdk::core::Control>();
        control->init();
        follow = std::make_unique<robo::sdk::examples::FollowPerson>(*control, targetImages, debugDir);
        app = std::thread([this] { follow->run(); });
    }

    // lets the simulation run until time t (bus clock, seconds)
    void runUntil(double t) const {
        while (world.time() < t) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // runs until the condition holds or the deadline passes; whether it held
    bool waitFor(const std::function<bool()>& condition, double deadline) const {
        while (world.time() < deadline) {
            if (condition()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return condition();
    }

    // like Ctrl+C: the loop ends, FollowPerson and Control are destroyed (Control hands
    // control back to the remote), then the world stops
    void shutdown() {
        robo::sdk::core::Control::stopRequested = true;
        if (app.joinable()) app.join();
        follow.reset();
        control.reset();
        world.stop();
    }

    bool running() const { return app.joinable(); }

private:
    std::unique_ptr<robo::sdk::core::Control> control;
    std::unique_ptr<robo::sdk::examples::FollowPerson> follow;
    std::thread app;
};
}

#endif //FOLLOW_TESTS_SIMRUN_H
