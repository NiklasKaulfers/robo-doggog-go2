// FollowPerson as a whole in the simulated world: does the robot walk after the person,
// around obstacles, and look for them when they are gone? Every scenario also checks the
// rules that always hold (SafetyRules.h). Each runs in real time, in its own process.
#include <algorithm>
#include <cmath>
#include "doctest/doctest.h"
#include "SafetyRules.h"
#include "SimRun.h"

using namespace sim;

TEST_CASE("the robot walks up to a person standing 2 m ahead and stops in front of them") {
    SimRun run;
    const int person = run.world.addWalker(Walker {2.0f, 0.0f});
    run.start();
    run.runUntil(8);
    // the following distance is 0.5 m from the camera
    CAPTURE(run.world.cameraDistanceTo(person));
    CHECK(run.world.cameraDistanceTo(person) > 0.4f);
    CHECK(run.world.cameraDistanceTo(person) < 0.7f);
    run.shutdown();
    checkSafetyRules(run.world);
}

TEST_CASE("the robot keeps up with a person walking away") {
    SimRun run;
    const int person = run.world.addWalker(Walker {1.5f, 0.0f, 0.4f, 0.0f});
    run.start();
    run.runUntil(10);
    CAPTURE(run.world.cameraDistanceTo(person));
    CHECK(run.world.cameraDistanceTo(person) > 0.4f);
    CHECK(run.world.cameraDistanceTo(person) < 1.6f);
    CHECK(run.world.pose().x > 2.5f);
    run.shutdown();
    checkSafetyRules(run.world);
}

TEST_CASE("the robot doesn't touch a low obstacle between it and the person") {
    SimRun run;
    // knee high: it doesn't hide the person from the camera, but it can't be stepped over
    run.world.addBox(Box {1.0f, -0.6f, 1.4f, 0.6f, 0.4f});
    run.world.addWalker(Walker {3.0f, 0.0f});
    run.start();
    run.runUntil(10);
    run.shutdown();
    // it turns to go around and comes close to the box, without touching it. Getting all
    // the way around is a known issue, see test_known_issues.cpp
    checkSafetyRules(run.world);
}

TEST_CASE("when the person leaves the view the robot turns and goes where they went and later roams") {
    SimRun run;
    const int person = run.world.addWalker(Walker {2.0f, 0.0f});
    run.start();
    run.runUntil(3);
    // they walk off to the left and disappear behind something
    run.world.setWalker(person, Walker {run.world.walker(person).x, run.world.walker(person).y, 0.0f, 1.0f});
    run.runUntil(4);
    Walker gone = run.world.walker(person);
    gone.hidden = true;
    gone.vy = 0;
    run.world.setWalker(person, gone);
    const double hidden = run.world.time();
    const Pose whenHidden = run.world.pose();
    run.runUntil(hidden + 4.0);
    const Pose searched = run.world.pose();
    run.runUntil(hidden + 13.0);
    run.shutdown();

    // turned left, toward where they were heading
    const auto searching = moves(hidden, hidden + 2.0);
    CHECK(std::any_of(searching.begin(), searching.end(), [](const Move& m) { return m.z > 0.1f; }));
    CHECK(searched.yaw > whenHidden.yaw + 0.3f);
    // still moving after 10 s without them: roaming
    const auto roaming = moves(hidden + 10.5, 1e9);
    CHECK(std::any_of(roaming.begin(), roaming.end(), [](const Move& m) { return !m.stop(); }));
    checkSafetyRules(run.world);
}

TEST_CASE("someone in other clothes is not followed while the person is gone") {
    SimRun run;
    const int alice = run.world.addWalker(Walker {2.0f, 0.3f, 0, 0, testutil::RED, testutil::JEANS});
    run.start();
    run.runUntil(4);
    // she disappears, he shows up front right
    Walker gone = run.world.walker(alice);
    gone.hidden = true;
    run.world.setWalker(alice, gone);
    const Pose robot = run.world.pose();
    const int bob = run.world.addWalker(Walker {robot.x + 1.5f, robot.y - 1.3f, 0, 0, testutil::GREEN, testutil::BLACK});
    const float startDistance = run.world.distanceTo(bob);
    run.runUntil(12);
    run.shutdown();

    // following him would bring the robot to about 0.8 m (center) from him
    CAPTURE(startDistance);
    CAPTURE(run.world.closestApproach(bob));
    CHECK(run.world.closestApproach(bob) > 1.3f);
    checkSafetyRules(run.world);
}
