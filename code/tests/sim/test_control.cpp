// Control: the robot connection, against the fake Unitree SDK. Remote controller input,
// the explore button combo, robot state age, movement through the obstacle avoidance
// service, stopping and handing control back.
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include "doctest/doctest.h"
#include <fakesdk/bus.hpp>
#include "core/Control.h"
#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/idl/go2/WirelessController_.hpp"

using robo::sdk::core::Control;
using fakesdk::bus;

TEST_CASE("init switches obstacle avoidance on when it is off") {
    bus().reset();
    Control control;
    control.init();
    CHECK(bus().count("avoid.SwitchSet") == 1);
    CHECK(bus().avoidance().enabled);
}

TEST_CASE("standing up succeeds when every SportClient call succeeds") {
    // the SDK returns 0 on success, an error code otherwise
    bus().reset();
    Control control;
    CHECK(control.standup());
    CHECK(bus().count("sport.RecoveryStand") == 1);
    CHECK(bus().count("sport.BalanceStand") == 1);
    CHECK(bus().count("sport.FreeWalk") == 1);
}

TEST_CASE("standing up fails when one SportClient call fails and still tries all of them") {
    bus().reset();
    bus().setResult("sport.BalanceStand", 3104);
    Control control;
    CHECK_FALSE(control.standup());
    CHECK(bus().count("sport.RecoveryStand") == 1);
    CHECK(bus().count("sport.FreeWalk") == 1);
}

namespace {
using unitree_go::msg::dds_::SportModeState_;
using unitree_go::msg::dds_::WirelessController_;

constexpr uint16_t R1 = 1 << 0;
constexpr uint16_t L1 = 1 << 1;
constexpr uint16_t Y = 1 << 11;
constexpr uint16_t A = 1 << 8;

void sendRemote(float lx, float ly, float rx, float ry, uint16_t keys) {
    WirelessController_ message;
    message.lx(lx);
    message.ly(ly);
    message.rx(rx);
    message.ry(ry);
    message.keys(keys);
    bus().publish("rt/wirelesscontroller", message);
}

void sendKeys(uint16_t keys) { sendRemote(0, 0, 0, 0, keys); }

// swaps a stream's buffer for the lifetime of the object
struct Capture {
    std::ostream& stream;
    std::ostringstream buffer;
    std::streambuf* old;
    explicit Capture(std::ostream& s) : stream(s), old(s.rdbuf(buffer.rdbuf())) {}
    ~Capture() { stream.rdbuf(old); }
    std::string str() const { return buffer.str(); }
};

size_t occurrences(const std::string& text, const std::string& what) {
    size_t n = 0;
    for (size_t pos = text.find(what); pos != std::string::npos; pos = text.find(what, pos + 1)) n++;
    return n;
}
}

TEST_CASE("init configures both clients in order and subscribes to both topics") {
    bus().reset();
    Control control;
    control.init();
    CHECK(bus().count("sport.SetTimeout") == 1);
    CHECK(bus().count("sport.Init") == 1);
    CHECK(bus().count("sport.WaitLeaseApplied") == 1);
    CHECK(bus().count("avoid.SetTimeout") == 1);
    CHECK(bus().count("avoid.Init") == 1);
    CHECK(bus().hasSubscriber("rt/sportmodestate"));
    CHECK(bus().hasSubscriber("rt/wirelesscontroller"));
}

TEST_CASE("init leaves obstacle avoidance alone when it is already on") {
    bus().reset();
    bus().setAvoidanceEnabled(true);
    Control control;
    control.init();
    CHECK(bus().count("avoid.SwitchSet") == 0);
    CHECK(bus().avoidance().enabled);
}

TEST_CASE("init switches avoidance on when reading its state fails") {
    bus().reset();
    bus().setResult("avoid.SwitchGet", 3104);
    bus().setAvoidanceEnabled(true);
    Control control;
    control.init();
    auto sets = bus().calls("avoid.SwitchSet");
    REQUIRE(sets.size() == 1);
    CHECK(sets[0].args[0] == 1.0f);
}

TEST_CASE("init survives a failing SwitchSet and reports it") {
    bus().reset();
    bus().setResult("avoid.SwitchSet", 3104);
    Control control;
    Capture err(std::cerr);
    CHECK_NOTHROW(control.init());
    CHECK(err.str().find("could not enable obstacle avoidance") != std::string::npos);
    CHECK_FALSE(bus().avoidance().enabled);
}

TEST_CASE("init never switches obstacle avoidance off") {
    for (int scenario = 0; scenario < 3; scenario++) {
        bus().reset();
        bus().setAvoidanceEnabled(scenario == 1);
        if (scenario == 2) bus().setResult("avoid.SwitchGet", 3104);
        Control control;
        control.init();
        for (const auto& call : bus().calls("avoid.SwitchSet"))
            CHECK(call.args[0] == 1.0f);
    }
}

TEST_CASE("state age is infinite before the first robot state") {
    bus().reset();
    Control control;
    control.init();
    CHECK(std::isinf(control.stateAge()));
}

TEST_CASE("state age is small after a robot state and grows with time") {
    bus().reset();
    Control control;
    control.init();
    bus().publish("rt/sportmodestate", SportModeState_());
    const float first = control.stateAge();
    CHECK(first >= 0.0f);
    CHECK(first < 0.1f);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    CHECK(control.stateAge() >= 0.14f);
    CHECK(control.stateAge() > first);
    // a new message makes it fresh again
    bus().publish("rt/sportmodestate", SportModeState_());
    CHECK(control.stateAge() < 0.1f);
}

TEST_CASE("position and orientation come from the last robot state") {
    bus().reset();
    Control control;
    control.init();
    SportModeState_ state;
    state.position() = {1.5f, -2.5f, 0.32f};
    state.imu_state().rpy() = {0.1f, -0.2f, 1.3f};
    bus().publish("rt/sportmodestate", state);
    CHECK(control.positionX() == doctest::Approx(1.5f));
    CHECK(control.positionY() == doctest::Approx(-2.5f));
    CHECK(control.positionZ() == doctest::Approx(0.32f));
    CHECK(control.roll() == doctest::Approx(0.1f));
    CHECK(control.pitch() == doctest::Approx(-0.2f));
    CHECK(control.yaw() == doctest::Approx(1.3f));
}

TEST_CASE("the remote is inactive before any input") {
    bus().reset();
    Control control;
    control.init();
    CHECK_FALSE(control.controllerActive());
}

TEST_CASE("stick noise below the dead zone does not count as remote use") {
    bus().reset();
    Control control;
    control.init();
    sendRemote(0.05f, -0.05f, 0.05f, -0.05f, 0);
    CHECK_FALSE(control.controllerActive());
    sendRemote(0, 0, 0, 0, 0);
    CHECK_FALSE(control.controllerActive());
}

TEST_CASE("a stick deflection on any axis makes the remote active") {
    for (int axis = 0; axis < 4; axis++) {
        for (float value : {0.2f, -0.2f}) {
            CAPTURE(axis);
            CAPTURE(value);
            bus().reset();
            Control control;
            control.init();
            float v[4] = {0, 0, 0, 0};
            v[axis] = value;
            sendRemote(v[0], v[1], v[2], v[3], 0);
            CHECK(control.controllerActive());
        }
    }
}

TEST_CASE("any button press makes the remote active") {
    bus().reset();
    Control control;
    control.init();
    sendKeys(A);
    CHECK(control.controllerActive());
}

TEST_CASE("the remote stays active for 1.5 seconds after the last input and then lapses") {
    bus().reset();
    Control control;
    control.init();
    sendRemote(0.5f, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    CHECK(control.controllerActive());
    // an idle message must not refresh the hold time
    sendRemote(0, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    CHECK_FALSE(control.controllerActive());
}

TEST_CASE("holding the explore combo over several messages is one press") {
    bus().reset();
    Control control;
    control.init();
    for (int i = 0; i < 5; i++) sendKeys(L1 | R1 | Y);
    CHECK(control.exploreComboPressed());
    CHECK_FALSE(control.exploreComboPressed());
}

TEST_CASE("releasing and pressing the explore combo again counts again") {
    bus().reset();
    Control control;
    control.init();
    sendKeys(L1 | R1 | Y);
    CHECK(control.exploreComboPressed());
    sendKeys(0);
    sendKeys(L1 | R1 | Y);
    CHECK(control.exploreComboPressed());
    CHECK_FALSE(control.exploreComboPressed());
}

TEST_CASE("two explore presses before the trick loop asks are both counted") {
    bus().reset();
    Control control;
    control.init();
    sendKeys(L1 | R1 | Y);
    sendKeys(0);
    sendKeys(L1 | R1 | Y);
    CHECK(control.exploreComboPressed());
    CHECK(control.exploreComboPressed());
    CHECK_FALSE(control.exploreComboPressed());
}

TEST_CASE("L1 and R1 without Y is not an explore press") {
    bus().reset();
    Control control;
    control.init();
    sendKeys(L1 | R1);
    CHECK_FALSE(control.exploreComboPressed());
    sendKeys(L1 | Y);
    sendKeys(R1 | Y);
    CHECK_FALSE(control.exploreComboPressed());
}

TEST_CASE("the explore combo with an extra button still counts") {
    bus().reset();
    Control control;
    control.init();
    sendKeys(L1 | R1 | Y | A);
    CHECK(control.exploreComboPressed());
}

TEST_CASE("the first move takes API control and then moves through avoidance") {
    bus().reset();
    Control control;
    control.init();
    control.moveRelative(0.3f, 0.1f);
    auto calls = bus().calls();
    // find the two calls after init in order
    std::vector<std::string> apis;
    for (const auto& c : calls)
        if (c.api == "avoid.UseRemoteCommandFromApi" || c.api == "avoid.Move") apis.push_back(c.api);
    REQUIRE(apis.size() == 2);
    CHECK(apis[0] == "avoid.UseRemoteCommandFromApi");
    CHECK(apis[1] == "avoid.Move");
    CHECK(bus().calls("avoid.UseRemoteCommandFromApi")[0].args[0] == 1.0f);
    auto moves = bus().calls("avoid.Move");
    REQUIRE(moves.size() == 1);
    CHECK(moves[0].args[0] == doctest::Approx(0.3f));
    CHECK(moves[0].args[1] == doctest::Approx(0.0f));
    CHECK(moves[0].args[2] == doctest::Approx(0.1f));
    CHECK(bus().count("sport.Move") == 0);
}

TEST_CASE("further moves do not take API control again") {
    bus().reset();
    Control control;
    control.init();
    control.moveRelative(0.3f, 0.1f);
    control.moveRelative(0.2f, 0.0f);
    control.moveForwardBackward(0.1f);
    CHECK(bus().count("avoid.UseRemoteCommandFromApi") == 1);
    CHECK(bus().count("avoid.Move") == 3);
}

TEST_CASE("each movement helper sends the matching avoidance move") {
    bus().reset();
    Control control;
    control.init();
    control.rotateLeft(0.5f);
    control.rotateRight(0.5f);
    control.moveForwardBackward(0.4f);
    control.moveRightLeft(0.3f);
    auto moves = bus().calls("avoid.Move");
    REQUIRE(moves.size() == 4);
    CHECK(moves[0].args == std::vector<float>{0.0f, 0.0f, 0.5f});
    CHECK(moves[1].args == std::vector<float>{0.0f, 0.0f, -0.5f});
    CHECK(moves[2].args == std::vector<float>{0.4f, 0.0f, 0.0f});
    CHECK(moves[3].args == std::vector<float>{0.0f, 0.3f, 0.0f});
    CHECK(bus().count("sport.Move") == 0);
}

TEST_CASE("a refused API control sends no move and never falls back to the SportClient") {
    bus().reset();
    bus().setResult("avoid.UseRemoteCommandFromApi", 3104);
    Control control;
    control.init();
    Capture err(std::cerr);
    control.moveRelative(0.3f, 0.1f);
    CHECK(bus().count("avoid.Move") == 0);
    CHECK(bus().count("sport.Move") == 0);
    // it asks again on the next move
    control.moveRelative(0.3f, 0.1f);
    CHECK(bus().count("avoid.UseRemoteCommandFromApi") == 2);
    CHECK(bus().count("avoid.Move") == 0);
    CHECK(bus().count("sport.Move") == 0);
}

TEST_CASE("a failing avoidance move does not fall back to the SportClient") {
    bus().reset();
    bus().setResult("avoid.Move", 3104);
    Control control;
    control.init();
    Capture err(std::cerr);
    control.moveRelative(0.3f, 0.1f);
    CHECK(bus().count("avoid.Move") == 1);
    CHECK(bus().count("sport.Move") == 0);
    CHECK(err.str().find("error in avoidance move") != std::string::npos);
}

TEST_CASE("stopping without API control returns true and sends nothing") {
    bus().reset();
    Control control;
    control.init();
    CHECK(control.stopMovement());
    CHECK(bus().count("avoid.Move") == 0);
    CHECK(bus().count("sport.Move") == 0);
    CHECK(bus().count("sport.StopMove") == 0);
}

TEST_CASE("stopping with API control sends a zero move") {
    bus().reset();
    Control control;
    control.init();
    control.moveForwardBackward(0.4f);
    CHECK(control.stopMovement());
    auto moves = bus().calls("avoid.Move");
    REQUIRE(moves.size() == 2);
    CHECK(moves[1].args == std::vector<float>{0.0f, 0.0f, 0.0f});
    CHECK(bus().avoidance().vx == 0.0f);
}

TEST_CASE("a failing stop returns false and is reported only once") {
    bus().reset();
    Control control;
    control.init();
    control.moveForwardBackward(0.4f);
    bus().setResult("avoid.Move", 3104);
    Capture err(std::cerr);
    CHECK_FALSE(control.stopMovement());
    CHECK_FALSE(control.stopMovement());
    CHECK_FALSE(control.stopMovement());
    CHECK(occurrences(err.str(), "stop command failed") == 1);
}

TEST_CASE("a stop failure is reported again after a success") {
    bus().reset();
    Control control;
    control.init();
    control.moveForwardBackward(0.4f);
    Capture err(std::cerr);
    bus().setResult("avoid.Move", 3104);
    CHECK_FALSE(control.stopMovement());
    bus().setResult("avoid.Move", 0);
    CHECK(control.stopMovement());
    bus().setResult("avoid.Move", 3104);
    CHECK_FALSE(control.stopMovement());
    CHECK(occurrences(err.str(), "stop command failed") == 2);
}

TEST_CASE("releasing control without having moved sends nothing") {
    bus().reset();
    Control control;
    control.init();
    control.releaseControl();
    CHECK(bus().count("avoid.Move") == 0);
    CHECK(bus().count("avoid.UseRemoteCommandFromApi") == 0);
}

TEST_CASE("releasing control stops then hands the robot back to the remote") {
    bus().reset();
    Control control;
    control.init();
    control.moveForwardBackward(0.4f);
    control.releaseControl();
    auto calls = bus().calls();
    REQUIRE(calls.size() >= 2);
    const auto& last = calls[calls.size() - 1];
    const auto& beforeLast = calls[calls.size() - 2];
    CHECK(beforeLast.api == "avoid.Move");
    CHECK(beforeLast.args == std::vector<float>{0.0f, 0.0f, 0.0f});
    CHECK(last.api == "avoid.UseRemoteCommandFromApi");
    CHECK(last.args[0] == 0.0f);
    CHECK_FALSE(bus().avoidance().apiControl);
    // a later move takes control again
    control.moveForwardBackward(0.4f);
    CHECK(bus().count("avoid.UseRemoteCommandFromApi") == 3);
    CHECK(bus().avoidance().apiControl);
}

TEST_CASE("destroying a Control that moved hands control back") {
    bus().reset();
    {
        Control control;
        control.init();
        control.moveForwardBackward(0.4f);
        CHECK(bus().avoidance().apiControl);
    }
    auto calls = bus().calls();
    REQUIRE(calls.size() >= 2);
    CHECK(calls[calls.size() - 2].api == "avoid.Move");
    CHECK(calls[calls.size() - 2].args == std::vector<float>{0.0f, 0.0f, 0.0f});
    CHECK(calls.back().api == "avoid.UseRemoteCommandFromApi");
    CHECK(calls.back().args[0] == 0.0f);
    CHECK_FALSE(bus().avoidance().apiControl);
    CHECK_FALSE(bus().hasSubscriber("rt/wirelesscontroller"));
}

TEST_CASE("destroying a Control that never moved sends no avoidance commands") {
    bus().reset();
    {
        Control control;
        control.init();
    }
    CHECK(bus().count("avoid.Move") == 0);
    CHECK(bus().count("avoid.UseRemoteCommandFromApi") == 0);
}

TEST_CASE("status shows the FSM mode name for a known mode") {
    bus().reset();
    Control control;
    control.init();
    SportModeState_ state;
    state.mode(1);
    bus().publish("rt/sportmodestate", state);
    Capture out(std::cout);
    control.printStatus();
    CHECK(out.str().find("STAND_DOWN") != std::string::npos);
}

TEST_CASE("status shows unknown for a mode outside the table") {
    bus().reset();
    Control control;
    control.init();
    SportModeState_ state;
    state.mode(200);
    bus().publish("rt/sportmodestate", state);
    Capture out(std::cout);
    control.printStatus();
    CHECK(out.str().find("UNKNOWN (200)") != std::string::npos);
}

TEST_CASE("hello sends the SportClient hello") {
    bus().reset();
    Control control;
    control.hello();
    CHECK(bus().count("sport.Hello") == 1);
}

TEST_CASE("sitting down reports success on 0 and failure on an error code") {
    bus().reset();
    Control control;
    CHECK(control.sitdown());
    CHECK(bus().count("sport.Sit") == 1);
    bus().setResult("sport.Sit", 3104);
    CHECK_FALSE(control.sitdown());
    CHECK(bus().count("sport.Sit") == 2);
}

TEST_CASE("no Control call ever sends a SportClient move") {
    bus().reset();
    bus().setResult("avoid.UseRemoteCommandFromApi", 3104);
    Control control;
    control.init();
    Capture err(std::cerr);
    control.rotateLeft(0.5f);
    bus().setResult("avoid.UseRemoteCommandFromApi", 0);
    control.rotateRight(0.5f);
    control.moveRightLeft(0.2f);
    control.moveRelative(0.3f, 0.1f);
    bus().setResult("avoid.Move", 3104);
    control.stopMovement();
    control.releaseControl();
    CHECK(bus().count("sport.Move") == 0);
}
