// What the simulated robot sees when app_sim (the app's main() with the fake SDK) runs,
// chosen with the FOLLOW_SIM_SCENARIO environment variable. The world starts when main()
// initializes DDS and presses Ctrl+C (SIGINT) after a while; at exit a summary of the
// calls the app made is printed for the command line tests to check:
//
//   follow         a person stands 2 m ahead, Ctrl+C after 5 s
//   standup_fails  the robot doesn't stand up (RecoveryStand fails)
//   idle           nobody around, Ctrl+C after 3 s (default)

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fakesdk/bus.hpp>
#include "SimWorld.h"

namespace {

// never deleted: it is stopped at exit, while the bus it uses still exists
sim::SimWorld* world = nullptr;
int person = -1;

void summary() {
    if (!world) return;
    world->stop();
    const auto calls = fakesdk::bus().calls();
    int moves = 0, forward = 0;
    std::string lastAvoid = "none", otherSport;
    for (const auto& call : calls) {
        if (call.api == "avoid.Move") {
            moves++;
            if (call.args.at(0) > 0) forward++;
        }
        if (call.api.rfind("avoid.", 0) == 0) {
            lastAvoid = call.api;
            for (const float a : call.args) lastAvoid += " " + std::to_string(static_cast<int>(a));
        }
        const bool sport = call.api.rfind("sport.", 0) == 0;
        const bool allowed = call.api == "sport.SetTimeout" || call.api == "sport.Init" ||
                             call.api == "sport.WaitLeaseApplied" || call.api == "sport.RecoveryStand" ||
                             call.api == "sport.BalanceStand" || call.api == "sport.FreeWalk";
        if (sport && !allowed) otherSport += " " + call.api;
    }
    std::cout << "[sim] movement commands: " << moves << std::endl;
    if (person >= 0) {
        std::cout << "[sim] moved toward the person: "
                  << (forward > 0 && world->distanceTo(person) < 1.8f ? "yes" : "no") << " (now "
                  << world->distanceTo(person) << " m away)" << std::endl;
    }
    // releasing control: the last command to the avoidance service gives it back to the remote
    std::cout << "[sim] last avoidance call: " << lastAvoid << std::endl;
    std::cout << "[sim] control handed back to the remote: "
              << (!fakesdk::bus().avoidance().apiControl && lastAvoid == "avoid.UseRemoteCommandFromApi 0" ? "yes" : "no")
              << std::endl;
    std::cout << "[sim] SportClient calls besides standing up: " << (otherSport.empty() ? "none" : otherSport) << std::endl;
    std::cout << "[sim] collision: " << (world->collided() ? world->collision() : "none") << std::endl;
}

void startScenario() {
    const char* chosen = std::getenv("FOLLOW_SIM_SCENARIO");
    const std::string scenario = chosen ? chosen : "idle";
    world = new sim::SimWorld();
    double stopAfter = 3;
    if (scenario == "follow") {
        person = world->addWalker(sim::Walker {2.0f, 0.0f});
        stopAfter = 5;
    } else if (scenario == "standup_fails") {
        fakesdk::bus().setResult("sport.RecoveryStand", 3104);
    } else if (scenario != "idle") {
        std::cerr << "[sim] unknown FOLLOW_SIM_SCENARIO " << scenario << std::endl;
        std::exit(2);
    }
    std::cout << "[sim] scenario " << scenario << std::endl;

    // Ctrl+C after a while, measured from when the app connects
    const double start = fakesdk::bus().now();
    world->setScript([start, stopAfter, raised = false](sim::SimWorld&, double t) mutable {
        if (!raised && t - start > stopAfter) {
            raised = true;
            std::raise(SIGINT);
        }
    });
    world->start();
    std::atexit(summary);
}

// main() initializes DDS before anything else talks to the robot
const bool registered = [] {
    fakesdk::bus().setOnDdsInit(startScenario);
    return true;
}();
}
