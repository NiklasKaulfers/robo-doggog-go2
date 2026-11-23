#ifndef FOLLOW_TESTS_SAFETYRULES_H
#define FOLLOW_TESTS_SAFETYRULES_H

// The rules every run of the robot code must keep, whatever happens in the scenario
// (see code/Agents.md): checked at the end of every simulation test.

#include <cmath>
#include <string>
#include "doctest/doctest.h"
#include <fakesdk/bus.hpp>
#include "SimWorld.h"

namespace sim {

// the follow loop's speed limits (FollowPerson.cpp and Explore.cpp clamp to these)
constexpr float MAX_FORWARD_SPEED = 0.6f;
constexpr float MAX_TURN_SPEED = 0.6f;

// SportClient calls that are fine: connecting, and standing up in main()
inline bool allowedSportCall(const std::string& api) {
    return api == "sport.SetTimeout" || api == "sport.Init" || api == "sport.WaitLeaseApplied" ||
           api == "sport.RecoveryStand" || api == "sport.BalanceStand" || api == "sport.FreeWalk";
}

inline void checkSafetyRules(const SimWorld& world, bool standingUpAllowed = false) {
    bool apiControl = false;
    for (const auto& call : fakesdk::bus().calls()) {
        CAPTURE(call.time);
        CAPTURE(call.api);
        // only the obstacle avoidance service moves the robot. Moving through the
        // SportClient would bypass the avoidance, and gaits, speed levels and the
        // avoidance mode are the remote controller's business
        if (call.api.rfind("sport.", 0) == 0) {
            const bool startup = call.api == "sport.SetTimeout" || call.api == "sport.Init" || call.api == "sport.WaitLeaseApplied";
            if (standingUpAllowed) CHECK_MESSAGE(allowedSportCall(call.api), "SportClient call not allowed: " << call.api);
            else CHECK_MESSAGE(startup, "the follow loop must not use the SportClient: " << call.api);
        }
        CHECK_MESSAGE(call.api != "avoid.MoveToAbsolutePosition", "position moves are not used");
        CHECK_MESSAGE(call.api != "avoid.MoveToIncrementPosition", "position moves are not used");
        // obstacle avoidance is never switched off
        if (call.api == "avoid.SwitchSet") CHECK_MESSAGE(call.args.at(0) != 0, "obstacle avoidance switched off");

        if (call.api == "avoid.UseRemoteCommandFromApi" && call.result == 0) apiControl = call.args.at(0) != 0;
        if (call.api == "avoid.Move") {
            // commands only while the avoidance service listens to the API, else they are lost
            CHECK_MESSAGE(apiControl, "Move sent while the avoidance service listens to the remote");
            CHECK(std::abs(call.args.at(0)) <= MAX_FORWARD_SPEED + 1e-6f);
            CHECK_MESSAGE(call.args.at(1) == 0, "sideways movement is never sent");
            CHECK(std::abs(call.args.at(2)) <= MAX_TURN_SPEED + 1e-6f);
        }
    }
    // the planning kept the robot clear of obstacles and people
    CHECK_MESSAGE(!world.collided(), "collision: " << world.collision());
}
}

#endif //FOLLOW_TESTS_SAFETYRULES_H
