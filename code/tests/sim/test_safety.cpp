// The safety rules of the follow app (code/Agents.md), checked in the simulated world:
// the remote always wins, the robot stops when its sensors fail, moves carefully on
// stairs, moves only through the obstacle avoidance service and hands control back when
// it ends. Every scenario also checks the rules that always hold (SafetyRules.h).
// Each runs in real time, in its own process.
#include <algorithm>
#include <cmath>
#include <functional>
#include "doctest/doctest.h"
#include "SafetyRules.h"
#include "SimRun.h"

using namespace sim;

namespace {

// a person ahead walking slowly away, so the robot keeps walking after them
int walkingAway(SimWorld& world, float x = 3.0f, float speed = 0.3f) {
    return world.addWalker(Walker {x, 0.0f, speed, 0.0f});
}

bool anyMoving(const std::vector<Move>& list) {
    return std::any_of(list.begin(), list.end(), [](const Move& m) { return !m.stop(); });
}

size_t stopCount(const std::vector<Move>& list) {
    return std::count_if(list.begin(), list.end(), [](const Move& m) { return m.stop(); });
}

// starts the run and waits until the robot follows the person
void startFollowing(SimRun& run) {
    run.start();
    REQUIRE_MESSAGE(run.waitFor([] { return anyMoving(moves()); }, 5), "the robot never started following");
    run.runUntil(run.world.time() + 1.0);
}

// breaks the camera or detector for 2 s while following: one stop, nothing else, then
// following again once it works
void checkStopsWhileBlind(const std::function<void(SimWorld&, bool)>& breakCamera) {
    SimRun run;
    walkingAway(run.world);
    startFollowing(run);

    const double broken = run.world.time();
    breakCamera(run.world, true);
    run.runUntil(broken + 2.0);
    const double repaired = run.world.time();
    breakCamera(run.world, false);
    run.runUntil(repaired + 3.0);
    run.shutdown();

    // stopped at the next loop tick, the one in progress may still have sent a command
    CHECK(stopCount(moves(broken, broken + 0.4)) >= 1);
    CHECK_FALSE(anyMoving(moves(broken + 0.25, repaired)));
    // the stop is sent once, repeating it would block the remote controller
    CHECK(stopCount(moves(broken, repaired)) == 1);
    CHECK_MESSAGE(anyMoving(moves(repaired, 1e9)), "the robot didn't follow again after the camera recovered");
    checkSafetyRules(run.world);
}
}

TEST_CASE("nobody in view means no movement commands at all") {
    SimRun run;
    run.start();
    run.runUntil(3);
    CHECK(moves().empty());
    CHECK(fakesdk::bus().count("avoid.UseRemoteCommandFromApi") == 0);
    run.shutdown();
    checkSafetyRules(run.world);
}

TEST_CASE("the remote controller takes over at once and the app waits until it is idle") {
    SimRun run;
    walkingAway(run.world);
    startFollowing(run);

    const double pressed = run.world.time();
    run.world.setRemote(Remote {0.0f, 0.5f});
    run.runUntil(pressed + 2.0);
    const double released = run.world.time();
    run.world.setRemote(Remote {});
    run.runUntil(released + 3.0);
    run.shutdown();

    // control goes back to the remote right away (within two loop ticks)
    const auto handedBack = callsTo("avoid.UseRemoteCommandFromApi", pressed, pressed + 0.4);
    REQUIRE_MESSAGE(!handedBack.empty(), "control was not handed to the remote");
    CHECK(handedBack.front().args.at(0) == 0);
    // nothing is sent while the remote is used and until it was idle for 1.5 s
    CHECK(moves(handedBack.front().time + 1e-6, released + 1.4).empty());
    CHECK(callsTo("avoid.UseRemoteCommandFromApi", handedBack.front().time + 1e-6, released + 1.4).empty());
    // then the app takes control again and follows on
    const auto takenBack = callsTo("avoid.UseRemoteCommandFromApi", released + 1.4, 1e9);
    REQUIRE_MESSAGE(!takenBack.empty(), "the app didn't continue after the remote was idle");
    CHECK(takenBack.front().args.at(0) == 1);
    CHECK(anyMoving(moves(released + 1.4, 1e9)));
    checkSafetyRules(run.world);
}

TEST_CASE("any button on the remote also takes over") {
    SimRun run;
    walkingAway(run.world);
    startFollowing(run);

    const double pressed = run.world.time();
    // a button the app doesn't use itself
    run.world.setRemote(Remote {0, 0, 0, 0, 1 << 8});
    run.runUntil(pressed + 1.0);
    run.world.setRemote(Remote {});
    run.runUntil(pressed + 1.5);
    run.shutdown();

    CHECK_FALSE(callsTo("avoid.UseRemoteCommandFromApi", pressed, pressed + 0.4).empty());
    CHECK_FALSE(anyMoving(moves(pressed + 0.4, pressed + 1.5)));
    checkSafetyRules(run.world);
}

TEST_CASE("a lost robot state stream stops the robot until it is back") {
    SimRun run;
    walkingAway(run.world);
    startFollowing(run);

    const double lost = run.world.time();
    run.world.stateStream = false;
    run.runUntil(lost + 2.0);
    const double back = run.world.time();
    run.world.stateStream = true;
    run.runUntil(back + 3.0);
    run.shutdown();

    // odometry older than 0.5 s is not trusted: a stop at the next tick after that
    CHECK(stopCount(moves(lost + 0.4, lost + 0.8)) >= 1);
    CHECK_FALSE(anyMoving(moves(lost + 0.7, back)));
    CHECK(stopCount(moves(lost, back)) == 1);
    CHECK_MESSAGE(anyMoving(moves(back, 1e9)), "the robot didn't follow again after the state came back");
    checkSafetyRules(run.world);
}

TEST_CASE("a camera error stops the robot until the camera works again") {
    checkStopsWhileBlind([](SimWorld& world, bool broken) { world.cameraError = broken ? 3104 : 0; });
}

TEST_CASE("a camera image that can't be decoded stops the robot") {
    checkStopsWhileBlind([](SimWorld& world, bool broken) { world.cameraGarbage = broken; });
}

TEST_CASE("a failing pose detector stops the robot") {
    checkStopsWhileBlind([](SimWorld& world, bool broken) { world.detectorFails = broken; });
}

TEST_CASE("with a stale height map the robot only turns and doesn't walk forward") {
    SimRun run;
    // to the left and walking sideways, so there is turning to do
    run.world.addWalker(Walker {2.5f, 0.8f, 0.0f, 0.3f});
    startFollowing(run);

    const double stale = run.world.time();
    run.world.mapStream = false;
    run.runUntil(stale + 1.3);
    const Pose before = run.world.pose();
    run.runUntil(stale + 4.0);
    const Pose after = run.world.pose();
    run.shutdown();

    // a height map older than 1 s doesn't show obstacles anymore: no forward speed
    for (const auto& m : moves(stale + 1.25, 1e9)) {
        CAPTURE(m.time);
        CHECK(m.x <= 0);
    }
    CHECK(std::hypot(after.x - before.x, after.y - before.y) < 0.05f);
    // still turning toward the person
    const auto turning = moves(stale + 1.25, 1e9);
    CHECK(std::any_of(turning.begin(), turning.end(), [](const Move& m) { return m.z != 0; }));
    checkSafetyRules(run.world);
}

TEST_CASE("on stairs the robot walks slowly and straight and never backwards and stops when it tilts sideways") {
    SimRun run;
    walkingAway(run.world, 3.0f, 0.5f);
    startFollowing(run);

    // nose up: climbing
    const double up = run.world.time();
    run.world.setTilt(0, -0.2f);
    run.runUntil(up + 2.0);
    // nose down: going down
    const double down = run.world.time();
    run.world.setTilt(0, 0.2f);
    run.runUntil(down + 2.0);
    // tilted sideways: about to slip off a step
    const double slipping = run.world.time();
    run.world.setTilt(0.4f, 0.2f);
    run.runUntil(slipping + 1.5);
    // level again
    const double level = run.world.time();
    run.world.setTilt(0, 0);
    run.runUntil(level + 3.0);
    run.shutdown();

    // stairs are detected after 0.3 s tilted
    for (const auto& m : moves(up + 0.5, down)) {
        CAPTURE(m.time);
        CHECK(m.x >= 0);
        CHECK(m.x <= 0.35f + 1e-6f);
        CHECK(std::abs(m.z) <= 0.2f + 1e-6f);
    }
    for (const auto& m : moves(down + 0.2, slipping)) {
        CAPTURE(m.time);
        CHECK(m.x >= 0);
        CHECK(m.x <= 0.25f + 1e-6f);
        CHECK(std::abs(m.z) <= 0.2f + 1e-6f);
    }
    CHECK(anyMoving(moves(up + 0.5, slipping)));
    CHECK(stopCount(moves(slipping, slipping + 0.35)) >= 1);
    CHECK_FALSE(anyMoving(moves(slipping + 0.25, level)));
    // 1 s on level ground ends the stair mode: full speed again
    const auto afterwards = moves(level + 1.3, 1e9);
    CHECK(std::any_of(afterwards.begin(), afterwards.end(), [](const Move& m) { return m.x > 0.35f; }));
    checkSafetyRules(run.world);
}

TEST_CASE("Ctrl+C stops the robot and hands control back to the remote") {
    SimRun run;
    walkingAway(run.world);
    startFollowing(run);

    const double interrupted = run.world.time();
    run.shutdown();

    std::vector<fakesdk::Call> avoidCalls;
    for (const auto& call : fakesdk::bus().calls())
        if (call.api.rfind("avoid.", 0) == 0 && call.time >= interrupted) avoidCalls.push_back(call);
    REQUIRE(!avoidCalls.empty());
    // the last word goes to the remote
    CHECK(avoidCalls.back().api == "avoid.UseRemoteCommandFromApi");
    CHECK(avoidCalls.back().args.at(0) == 0);
    REQUIRE(!moves(interrupted, 1e9).empty());
    CHECK(moves(interrupted, 1e9).back().stop());
    // the tick in progress may finish, nothing moves after that
    CHECK_FALSE(anyMoving(moves(interrupted + 0.15, 1e9)));
    CHECK_FALSE(fakesdk::bus().avoidance().apiControl);
    checkSafetyRules(run.world);
}

TEST_CASE("the explore combo toggles roaming once per press and the remote keeps priority meanwhile") {
    SimRun run;
    run.start();
    run.runUntil(1.0);
    const Pose start = run.world.pose();

    // L1 + R1 + Y held for a second is one press
    const double pressed = run.world.time();
    run.world.setRemote(Remote {0, 0, 0, 0, KEY_L1 | KEY_R1 | KEY_Y});
    run.runUntil(pressed + 1.0);
    const double released = run.world.time();
    run.world.setRemote(Remote {});
    run.runUntil(released + 6.0);
    const Pose roamed = run.world.pose();

    // pressed again: back to following, and nobody is there to follow
    const double again = run.world.time();
    run.world.setRemote(Remote {0, 0, 0, 0, KEY_L1 | KEY_R1 | KEY_Y});
    run.runUntil(again + 0.5);
    const double releasedAgain = run.world.time();
    run.world.setRemote(Remote {});
    run.runUntil(releasedAgain + 3.5);
    run.shutdown();

    // nothing while the buttons are held and the remote has priority
    CHECK_FALSE(anyMoving(moves(pressed, released + 1.4)));
    // then it roams on its own, nobody needed
    CHECK(anyMoving(moves(released + 1.4, again)));
    CHECK(std::hypot(roamed.x - start.x, roamed.y - start.y) > 0.3f);
    // the second press ends it
    CHECK_FALSE(anyMoving(moves(releasedAgain + 1.7, 1e9)));
    checkSafetyRules(run.world);
}
