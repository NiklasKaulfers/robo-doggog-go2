// Known issues: scenarios the robot code doesn't handle yet. They are not part of the
// pre-install gate (label "known"); run them with `ctest -L known`. When one starts to
// pass, move it into test_follow.cpp or test_safety.cpp.
#include "doctest/doctest.h"
#include "SafetyRules.h"
#include "SimRun.h"

using namespace sim;

// The robot gets stuck next to the corner of the box, facing along its planned route
// but not walking: the route planner (ObstacleMap::planPath) lets the route pass obstacles
// 0.3 m from the robot's center (INFLATE_CELLS), while the forward speed in FollowPerson
// needs MIN_CLEARANCE (0.35 m) free along rays that start 0.3 m ahead at the center and
// 0.2 m to each side (ObstacleMap::freeDistance). Next to a corner the side ray hits the
// box at once, so x is clamped to 0, and as the heading already points along the route the
// robot doesn't turn either. It stays there as long as the person stays behind the box.
TEST_CASE("the robot finds its way around a low obstacle to the person") {
    SimRun run;
    run.world.addBox(Box {1.0f, -0.6f, 1.4f, 0.6f, 0.4f});
    const int person = run.world.addWalker(Walker {3.0f, 0.0f});
    run.start();
    run.runUntil(15);
    CAPTURE(run.world.pose().x);
    CAPTURE(run.world.pose().y);
    CHECK(run.world.cameraDistanceTo(person) < 1.0f);
    run.shutdown();
    checkSafetyRules(run.world);
}
