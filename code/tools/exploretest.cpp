// Self-test of the ExplorePlanner with a simulated robot on synthetic terrain: goal
// choice and scoring, failed goals, timeouts, look-around, stopping when the map is
// missing / stale / in another frame, and the safety limits, checked on every simulated
// step. Runs anywhere, needs neither the robot nor the models:
//
//   cd code && ./build-tools/exploretest
//
// Exit code 0 when all checks pass.

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "../src/core/ExplorePlanner.h"
#include "../src/core/ObstacleMap.h"

using robo::sdk::core::ExploreCommand;
using robo::sdk::core::ExploreConfig;
using robo::sdk::core::ExplorePlanner;
using robo::sdk::core::ExplorePose;
using robo::sdk::core::HeightMapFrame;
using robo::sdk::core::ObstacleMap;
using State = ExplorePlanner::State;
// terrain height at a world point, NAN where the LiDAR sees nothing
using Terrain = std::function<float(float, float)>;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok    " : "  FAIL  ") << what << std::endl;
    if (!condition) failures++;
}

// a map with a fake clock that the test moves forward by hand
struct TestMap {
    ObstacleMap map;
    float time = 0;

    TestMap() { map.setClock([this] { return time; }); }

    // one LiDAR frame of the given terrain, centered on (cx, cy)
    void observe(const Terrain& terrain, float cx = 0, float cy = 0, float half = 5.0f, float res = 0.1f) {
        const int n = static_cast<int>(std::lround(2 * half / res));
        std::vector<float> data(static_cast<size_t>(n) * n);
        HeightMapFrame frame;
        frame.resolution = res;
        frame.width = frame.height = n;
        frame.originX = cx - n * res / 2;
        frame.originY = cy - n * res / 2;
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                const float h = terrain(frame.originX + (x + 0.5f) * res, frame.originY + (y + 0.5f) * res);
                // like the Go2: unobserved cells are filled with a huge value
                data[y * n + x] = std::isnan(h) ? 1e9f : h;
            }
        }
        frame.data = data.data();
        frame.size = data.size();
        map.integrate(frame);
    }
};

static float flat(float, float) { return 0; }

static bool inBox(float x, float y, float x0, float x1, float y0, float y1) {
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

// a simulated robot (unicycle) exploring terrain. Every tick advances the fake clock,
// shows the planner a fresh height map around the robot, steps it and moves the robot
// by the command. The safety limits are checked on every tick
struct Sim {
    TestMap t;
    Terrain terrain;
    ExplorePlanner planner;
    std::vector<std::string> logs;
    ExplorePose pose;
    ExploreCommand last;
    bool move = true;
    // robot body radius used for collision detection
    float bodyRadius = 0.15f;
    int ticks = 0;
    int limitViolations = 0;
    int collisions = 0;
    int reached = 0, reachedNotLooking = 0;
    float maxForward = 0;

    Sim(Terrain newTerrain, uint32_t seed, ExploreConfig cfg = {}, float x = 0, float y = 0, float yaw = 0)
        : terrain(std::move(newTerrain)), planner(t.map, seed, [this] { return t.time; }, cfg) {
        pose = {x, y, yaw};
        planner.setLogger([this](const std::string& message) { logs.push_back(message); });
    }

    bool collides() const {
        if (terrain(pose.x, pose.y) > 0.5f) return true;
        for (int i = 0; i < 8; i++) {
            const float a = i * 3.14159265f / 4;
            if (terrain(pose.x + bodyRadius * std::cos(a), pose.y + bodyRadius * std::sin(a)) > 0.5f) return true;
        }
        return false;
    }

    ExploreCommand tick(float dt = 0.1f, bool observe = true) {
        t.time += dt;
        if (observe) t.observe(terrain, pose.x, pose.y);
        const bool atGoal = observe && planner.hasGoal() &&
                            std::hypot(planner.goal().first - pose.x, planner.goal().second - pose.y) < planner.config().goalReached;
        last = planner.step(pose);
        ticks++;
        if (!(last.x >= 0 && last.x <= 0.4f && std::abs(last.z) <= 0.6f)) limitViolations++;
        maxForward = std::max(maxForward, last.x);
        if (atGoal) {
            reached++;
            if (planner.state() != State::LookingAround) reachedNotLooking++;
        }
        if (move && std::isfinite(last.x) && std::isfinite(last.z)) {
            pose.yaw += last.z * dt;
            pose.x += last.x * std::cos(pose.yaw) * dt;
            pose.y += last.x * std::sin(pose.yaw) * dt;
        }
        if (collides()) collisions++;
        return last;
    }

    void run(float seconds, float dt = 0.1f) {
        const int n = static_cast<int>(std::lround(seconds / dt));
        for (int i = 0; i < n; i++) tick(dt);
    }

    bool logged(const std::string& part) const {
        return std::any_of(logs.begin(), logs.end(), [&](const auto& l) { return l.find(part) != std::string::npos; });
    }
};

static std::string num(float v) { return std::to_string(v); }

// a 10 x 10 m room with 0.3m walls, nothing known outside of it
static float room(float x, float y) {
    if (std::abs(x) > 5 || std::abs(y) > 5) return NAN;
    return (std::abs(x) > 4.7f || std::abs(y) > 4.7f) ? 1.0f : 0.0f;
}

static void testNoMap() {
    std::cout << "no height map" << std::endl;
    Sim s(flat, 1);
    const ExploreCommand c = s.tick(0.1f, false);
    check(s.planner.state() == State::NoMap, "state is NoMap");
    check(c.x == 0 && std::abs(c.z) > 0, "looks around without walking");
    s.run(0.5f);
    check(s.planner.state() != State::NoMap, "walks (or looks) once the map arrives");
    s.planner.reset();
    check(s.planner.state() == State::NoMap, "reset() starts over at NoMap");
}

static void testStale() {
    std::cout << "stale height map" << std::endl;
    Sim s(flat, 1);
    s.run(1.0f);
    check(s.planner.state() == State::Walking || s.planner.state() == State::LookingAround, "exploring with a fresh map");
    const ExploreCommand c = s.tick(2.0f, false);
    check(s.planner.state() == State::MapStale && c.x == 0 && c.z == 0, "stops when the map is 2s old");
    check(s.logged("stale"), "stale map is logged");
    s.tick(0.1f, false);
    const size_t logCount = s.logs.size();
    s.tick(0.1f, false);
    check(s.logs.size() == logCount, "stale is logged once, not every step");
    s.tick(0.1f, true);
    check(s.planner.state() != State::MapStale, "recovers when the map updates again");
    check(s.logged("updating again"), "recovery is logged");

    // reset() forgets that the map was stale: no recovery message for a map that is fine
    s.tick(2.0f, false);
    s.planner.reset();
    s.logs.clear();
    s.tick(0.1f, true);
    check(!s.logged("updating again"), "reset() clears the stale flag");
}

static void testFrameMismatch() {
    std::cout << "frame mismatch" << std::endl;
    Sim s(flat, 1);
    s.run(0.5f);
    // odometry says the robot is 6m away from where the height map is centered
    s.pose.x = 6;
    s.t.time += 0.1f;
    s.t.observe(flat, 0, 0);
    ExploreCommand c = s.planner.step(s.pose);
    check(s.planner.state() == State::FrameMismatch && c.x == 0 && c.z == 0, "stops when the frames differ");
    s.logs.clear();
    for (int i = 0; i < 5; i++) {
        s.t.time += 0.1f;
        s.t.observe(flat, 0, 0);
        s.planner.step(s.pose);
    }
    check(s.logs.empty(), "mismatch is logged on the transition only");
    s.pose.x = 0.5f;
    s.t.time += 0.1f;
    s.t.observe(flat, 0, 0);
    c = s.planner.step(s.pose);
    check(s.planner.state() != State::FrameMismatch, "recovers when the frames match again");
    check(s.logged("match again"), "recovery is logged");
}

static void testInvalidPose() {
    std::cout << "invalid pose" << std::endl;
    const float bad[] = {NAN, INFINITY};
    for (const float b : bad) {
        for (int field = 0; field < 3; field++) {
            Sim s(flat, 1);
            s.run(0.5f);
            ExplorePose p = s.pose;
            (field == 0 ? p.x : field == 1 ? p.y : p.yaw) = b;
            const ExploreCommand c = s.planner.step(p);
            check(s.planner.state() == State::InvalidPose && c.x == 0 && c.z == 0,
                  "pose field " + std::to_string(field) + " = " + num(b) + " stops");
        }
    }
}

static void testDeterminism() {
    std::cout << "determinism" << std::endl;
    const auto finalPose = [](uint32_t seed) {
        Sim s(room, seed);
        s.run(40);
        return s.pose;
    };
    const ExplorePose a = finalPose(7), b = finalPose(7);
    check(a.x == b.x && a.y == b.y && a.yaw == b.yaw, "same seed, same path");
    bool differs = false;
    for (uint32_t seed = 8; seed < 12; seed++) {
        const ExplorePose c = finalPose(seed);
        differs = differs || c.x != a.x || c.y != a.y;
    }
    check(differs, "other seeds wander elsewhere");
}

static void testGoalChoice() {
    std::cout << "goal choice" << std::endl;
    bool inRange = true, all = true;
    for (uint32_t seed = 1; seed <= 20; seed++) {
        TestMap t;
        t.observe(flat);
        ExplorePlanner p(t.map, seed, [&t] { return t.time; });
        p.setLogger([](const std::string&) {});
        all = all && p.chooseGoal(0, 0);
        const float d = std::hypot(p.goal().first, p.goal().second);
        inRange = inRange && d >= 1.5f && d <= 8.0f;
    }
    check(all, "a goal is found on free terrain");
    check(inRange, "goals are 1.5 - 8m away");

    // nothing reachable: robot outside the known area
    TestMap t;
    t.observe(flat);
    ExplorePlanner p(t.map, 1, [&t] { return t.time; });
    p.setLogger([](const std::string&) {});
    check(!p.chooseGoal(50, 50), "no goal outside the known map");
}

static void testScoring() {
    std::cout << "scoring" << std::endl;
    TestMap t;
    t.observe(flat);
    ExplorePlanner p(t.map, 1, [&t] { return t.time; });
    p.setLogger([](const std::string&) {});
    const ObstacleMap::Spot plain {3, 0, false}, edge {3, 0, true};
    check(std::abs(p.scoreSpot(plain, 0) - 4.0f) < 1e-4f, "empty trail: novelty is capped at 4");
    check(std::abs(p.scoreSpot(edge, 0) - 6.0f) < 1e-4f, "frontier spots get +2");
    check(std::abs(p.scoreSpot(plain, 1.5f) - 5.5f) < 1e-4f, "random part is added");
    t.time += 0.1f;
    p.step({0, 0, 0});
    check(p.trail().size() == 1, "the first step records the start in the trail");
    check(std::abs(p.scoreSpot({1, 0, false}, 0) - 1.0f) < 1e-4f, "spots near the trail score by their distance to it");
    check(std::abs(p.scoreSpot({0.2f, 0.3f, false}, 0) - std::hypot(0.2f, 0.3f)) < 1e-4f, "novelty is the distance to the closest trail point");
    check(std::abs(p.scoreSpot({6, 0, false}, 0) - 4.0f) < 1e-4f, "novelty stays capped far from the trail");
    check(p.scoreSpot({6, 0, true}, 0) > p.scoreSpot({1.6f, 0, true}, 0), "far spots beat near ones");
}

static void testTimeoutAndFailedGoals() {
    std::cout << "goal timeout and failed goals" << std::endl;
    {
        // a robot that doesn't move never gets there
        Sim s(flat, 3);
        s.move = false;
        s.run(29);
        check(s.planner.hasGoal() && s.planner.failedGoals().empty(), "goal still pursued after 29s of walking");
        const auto firstGoal = s.planner.goal();
        s.run(3);
        check(s.planner.failedGoals().size() == 1, "goal given up after ~30s of walking");
        check(s.planner.failedGoals().front() == firstGoal, "the unreached goal is the failed one");
        check(s.planner.hasGoal() && s.planner.goal() != firstGoal, "another goal is picked");
        check(s.logged("not reached in time"), "timeout is logged");
        check(s.limitViolations == 0, "limits respected");
    }
    {
        // time away from walking doesn't count: a 60s gap between two steps adds at most maxTickGap
        Sim s(flat, 3);
        s.move = false;
        s.run(5);
        const float before = s.planner.goalElapsed();
        const auto goal = s.planner.goal();
        s.tick(60.0f);
        check(s.planner.goalElapsed() - before <= s.planner.config().maxTickGap + 1e-3f, "a 60s tick gap adds at most 0.5s to the goal time");
        check(s.planner.hasGoal() && s.planner.goal() == goal && s.planner.failedGoals().empty(), "goal survives a 60s tick gap");
    }
    {
        // time while stopped (stale map) doesn't count either
        Sim s(flat, 3);
        s.move = false;
        s.run(5);
        s.tick(2.0f, false);
        const float before = s.planner.goalElapsed();
        for (int i = 0; i < 50; i++) s.tick(1.0f, false);
        check(std::abs(s.planner.goalElapsed() - before) < 1e-4f, "50s with a stale map don't count");
        s.tick(0.1f);
        s.tick(0.1f);
        check(s.planner.failedGoals().empty(), "goal survives a stale period");
    }
    {
        // failed goals: spots near them are skipped, at most 20 are remembered
        Sim s(flat, 5);
        s.move = false;
        std::pair<float, float> seen {NAN, NAN};
        bool farEnough = true;
        size_t maxSize = 0;
        for (int i = 0; i < 1600; i++) {
            s.tick(0.5f);
            if (s.planner.hasGoal() && s.planner.goal() != seen) {
                seen = s.planner.goal();
                for (const auto& f : s.planner.failedGoals())
                    farEnough = farEnough && std::hypot(f.first - seen.first, f.second - seen.second) >= 1.0f;
            }
            maxSize = std::max(maxSize, s.planner.failedGoals().size());
        }
        check(farEnough, "new goals are at least 1m from every failed goal");
        check(s.planner.failedGoals().size() == 20 && maxSize == 20, "only the last 20 failed goals are kept");
    }
}

static void testPruning() {
    std::cout << "failed goals when everything is excluded" << std::endl;
    {
        // a small ring of reachable spots and a huge exclusion radius: every failure
        // excludes all of them
        ExploreConfig cfg;
        cfg.minGoalDistance = 1.5f;
        cfg.maxGoalDistance = 2.0f;
        cfg.failedGoalRadius = 3.0f;
        Sim s(flat, 2, cfg);
        s.move = false;
        bool pruned = false, partial = false, lookedAround = true, clearedAll = false;
        size_t before = 0;
        for (int i = 0; i < 1500; i++) {
            before = s.planner.failedGoals().size();
            s.tick(0.5f);
            const size_t after = s.planner.failedGoals().size();
            if (after < before + 1 && s.planner.state() == State::LookingAround && after + (before + 2) / 2 == before + 1) {
                pruned = true;
                partial = partial || (before + 1 >= 2 && after >= 1);
                lookedAround = lookedAround && std::abs(s.last.z) > 0 && s.last.x == 0;
            }
            if (before >= 2 && after == 0) clearedAll = true;
        }
        check(pruned, "when all spots are excluded the oldest half of the failed goals is dropped");
        check(partial, "...and not all of them (more than one failed goal)");
        check(!clearedAll, "failed goals are never wiped in one go");
        check(lookedAround, "the robot looks around meanwhile");
    }
    {
        // nothing reachable at all (boxed in): the failed goals stay
        Sim s(flat, 3);
        s.move = false;
        s.run(31);
        check(s.planner.failedGoals().size() == 1, "one failed goal to start with");
        s.terrain = [](float x, float y) {
            const float r = std::hypot(x, y);
            return (r >= 0.8f && r <= 1.1f) ? 1.0f : 0.0f;
        };
        // the current goal lies outside of the ring: its route fails, then no spots remain
        float maxX = 0;
        for (int i = 0; i < 100; i++) {
            s.tick(0.1f);
            maxX = std::max(maxX, s.last.x);
        }
        check(s.planner.failedGoals().size() == 2, "no spots at all: failed goals stay intact (2 after the route failure)");
        check(s.planner.state() == State::LookingAround && maxX == 0, "boxed in: looks around, never walks");
    }
}

static void testBoxedIn() {
    std::cout << "boxed in" << std::endl;
    Sim s([](float x, float y) {
        const float r = std::hypot(x, y);
        return (r >= 0.8f && r <= 1.1f) ? 1.0f : 0.0f;
    }, 4);
    float maxX = 0;
    bool looked = false;
    for (int i = 0; i < 600; i++) {
        s.tick();
        maxX = std::max(maxX, s.last.x);
        looked = looked || std::abs(s.last.z) > 0;
    }
    check(maxX == 0, "never walks forward");
    check(looked, "looks around instead");
    check(s.collisions == 0 && s.limitViolations == 0, "no collisions, limits respected");
}

static void testRegressionReached() {
    std::cout << "regression: 0.52m from the goal counts as arrived" << std::endl;
    int arrived = 0, runs = 0;
    for (uint32_t seed = 1; seed <= 6; seed++) {
        TestMap t;
        t.observe(flat);
        ExplorePlanner p(t.map, seed, [&t] { return t.time; });
        p.setLogger([](const std::string&) {});
        t.time += 0.1f;
        p.step({0, 0, 0});
        if (!p.hasGoal()) continue;
        runs++;
        const auto g = p.goal();
        const ExplorePose near {g.first - 0.52f, g.second, 0};
        t.time += 0.1f;
        t.observe(flat, near.x, near.y);
        p.step(near);
        if (!p.hasGoal() && p.failedGoals().empty() && p.state() == State::LookingAround) arrived++;
    }
    check(runs == 6 && arrived == runs, "a route of one cell is arrival, not a failure (" + std::to_string(arrived) + "/" + std::to_string(runs) + ")");
}

static void testWallBetween() {
    std::cout << "wall between robot and goal" << std::endl;
    // wall at x = 1..1.3 from y = -4 to 4, the way around is at the ends
    const Terrain wall = [](float x, float y) { return inBox(x, y, 1.0f, 1.3f, -4.0f, 4.0f) ? 1.0f : 0.0f; };
    int crossed = 0, collisions = 0, violations = 0, reached = 0, badReach = 0;
    for (uint32_t seed = 1; seed <= 8; seed++) {
        Sim s(wall, seed, {}, -2, 0);
        bool across = false;
        for (int i = 0; i < 1500; i++) {
            s.tick();
            if (s.planner.hasGoal() && s.planner.goal().first > 1.5f) across = true;
        }
        crossed += across;
        collisions += s.collisions;
        violations += s.limitViolations;
        reached += s.reached;
        badReach += s.reachedNotLooking;
    }
    check(crossed > 0, "goals behind the wall were chosen (" + std::to_string(crossed) + "/8 runs)");
    check(collisions == 0, "never entered the wall");
    check(violations == 0, "limits respected on every tick");
    check(reached > 0 && badReach == 0, "reaching a goal is followed by looking around (" + std::to_string(reached) + " arrivals)");
}

static void testPlanFailures() {
    std::cout << "consecutive route failures" << std::endl;
    TestMap t;
    bool ring = false;
    const Terrain ringTerrain = [](float x, float y) {
        const float r = std::hypot(x, y);
        return (r >= 1.0f && r <= 1.3f) ? 1.0f : 0.0f;
    };
    const Terrain terrain = [&ring, &ringTerrain](float x, float y) { return ring ? ringTerrain(x, y) : 0.0f; };
    ExploreConfig cfg;
    ExplorePlanner p(t.map, 9, [&t] { return t.time; }, cfg);
    // as soon as a goal is chosen the robot gets walled in, so its route can't be planned
    p.setLogger([&](const std::string& m) {
        if (m.find("walking to") == 0) {
            ring = true;
            t.observe(terrain);
        }
    });
    std::vector<State> states;
    for (int i = 0; i < 5; i++) {
        ring = false;
        t.time += 0.1f;
        t.observe(terrain);
        const ExploreCommand c = p.step({0, 0, 0});
        states.push_back(p.state());
        check(c.x == 0, "tick " + std::to_string(i) + ": no forward motion");
    }
    check(states[0] == State::Searching && states[3] == State::Searching, "first 4 failures: searching, standing still");
    check(states[4] == State::LookingAround, "5th failure in a row: looks around");
    check(p.failedGoals().size() == 5, "every failed route marks its goal");
    // the look-around lasts for a while, the counter starts from zero afterwards
    ring = false;
    t.time += 0.1f;
    t.observe(terrain);
    check(p.step({0, 0, 0}).z != 0 && p.state() == State::LookingAround, "keeps looking around");
    t.time += 2.5f;
    t.observe(terrain);
    p.step({0, 0, 0});
    check(p.state() == State::Searching, "after the look-around, one failure is just a failure again");
}

static void testTrail() {
    std::cout << "trail" << std::endl;
    ExploreConfig cfg;
    cfg.maxTrail = 10;
    Sim s(flat, 1, cfg);
    s.move = false;
    // drag the robot along a line by hand
    for (int i = 0; i < 300; i++) {
        s.pose.x = 0.1f * i;
        s.tick(0.1f);
    }
    const auto& trail = s.planner.trail();
    check(trail.size() == 10, "trail is capped at maxTrail (" + std::to_string(trail.size()) + ")");
    bool spaced = true;
    for (size_t i = 1; i < trail.size(); i++)
        spaced = spaced && std::hypot(trail[i].first - trail[i - 1].first, trail[i].second - trail[i - 1].second) >= 0.5f - 1e-3f;
    check(spaced, "trail points are at least 0.5m apart");
    check(std::abs(trail.back().first - 29.9f) < 0.6f, "newest points are kept, oldest dropped");

    s.planner.reset();
    check(s.planner.trail().empty() && s.planner.failedGoals().empty() && !s.planner.hasGoal() && s.planner.goalElapsed() == 0,
          "reset() forgets trail, failed goals and goal");
}

static void testBlockedAhead() {
    std::cout << "obstacle right ahead" << std::endl;
    Sim s([](float x, float y) { return inBox(x, y, 0.3f, 0.6f, -1.5f, 1.5f) ? 1.0f : 0.0f; }, 6);
    s.move = false;
    float maxX = 0;
    for (int i = 0; i < 300; i++) {
        s.tick();
        maxX = std::max(maxX, s.last.x);
    }
    check(maxX == 0, "obstacle 0.3m ahead: forward speed 0");
    check(s.limitViolations == 0, "limits respected");

    // a clear route, facing a wall a bit further away: slower than full speed
    Sim w([](float x, float y) { return inBox(x, y, 0.8f, 1.1f, -1.5f, 1.5f) ? 1.0f : 0.0f; }, 6);
    w.move = false;
    bool slowed = true;
    for (int i = 0; i < 300; i++) {
        w.tick();
        if (w.planner.state() == State::Walking) slowed = slowed && w.last.x <= std::max(0.0f, 0.5f * (0.5f - 0.35f)) + 0.01f;
    }
    check(slowed, "wall 0.8m ahead: speed capped to half the room left");
}

static void testTurnSign() {
    std::cout << "turning near +-pi" << std::endl;
    TestMap t;
    t.observe(flat);
    ExplorePlanner p(t.map, 5, [&t] { return t.time; });
    p.setLogger([](const std::string&) {});
    t.time += 0.1f;
    p.step({0, 0, 0});
    check(p.hasGoal(), "goal chosen");
    const auto g = p.goal();
    const float a = std::atan2(g.second, g.first);
    int positive = 0, negative = 0;
    for (int i = 0; i < 30; i++) {
        // the goal is almost exactly behind the robot, the heading error jumps between +-pi
        const float eps = (i % 2 == 0) ? 0.05f : -0.05f;
        const ExplorePose pose {0, 0, ExplorePlanner::wrapAngle(a - 3.14159265f + eps)};
        t.time += 0.1f;
        t.observe(flat);
        const ExploreCommand c = p.step(pose);
        positive += c.z > 0;
        negative += c.z < 0;
        check(std::abs(c.z) == 0.6f && c.x == 0, "tick " + std::to_string(i) + ": full turn, no forward motion");
    }
    check(positive == 30 || negative == 30, "turn direction stays the same (+" + std::to_string(positive) + "/-" + std::to_string(negative) + ")");

    // away from +-pi the turn follows the heading error
    const ExplorePose left {0, 0, ExplorePlanner::wrapAngle(a - 1.0f)}, right {0, 0, ExplorePlanner::wrapAngle(a + 1.0f)};
    t.time += 0.1f;
    t.observe(flat);
    check(p.step(left).z > 0, "goal to the left: turns left");
    t.time += 0.1f;
    t.observe(flat);
    check(p.step(right).z < 0, "goal to the right: turns right");
    t.time += 0.1f;
    t.observe(flat);
    const ExploreCommand ahead = p.step({0, 0, a});
    check(ahead.x > 0 && ahead.x <= 0.4f && std::abs(ahead.z) < 0.1f, "facing the goal: walks straight");
}

static void testConfigLimits() {
    std::cout << "configured limits" << std::endl;
    ExploreConfig cfg;
    cfg.exploreSpeed = 5;
    cfg.lookAroundSpeed = 5;
    TestMap t;
    ExplorePlanner p(t.map, 1, [&t] { return t.time; }, cfg);
    p.setLogger([](const std::string&) {});
    check(p.config().exploreSpeed == 0.4f && p.config().lookAroundSpeed <= 0.6f, "speeds can't be configured above the limits");
    const ExploreCommand c = p.step({0, 0, 0});
    check(c.x == 0 && std::abs(c.z) <= 0.6f, "look-around within limits");
    ExplorePlanner d(t.map, 1);
    check(ExplorePlanner::wrapAngle(7.0f) < 3.15f && std::abs(ExplorePlanner::wrapAngle(7.0f) - (7.0f - 2 * 3.14159265f)) < 1e-4f, "wrapAngle");
    // default clock and logger work
    d.step({0, 0, 0});
    check(d.state() == State::NoMap, "default clock and logger");
}

static void testLongRun() {
    std::cout << "long run in a walled room" << std::endl;
    for (uint32_t seed = 1; seed <= 2; seed++) {
        Sim s(room, seed);
        std::set<std::pair<int, int>> cells;
        size_t early = 0;
        float forward = 0;
        for (int i = 0; i < 3000; i++) {
            s.tick();
            cells.emplace(static_cast<int>(std::floor(s.pose.x)), static_cast<int>(std::floor(s.pose.y)));
            forward += s.last.x * 0.1f;
            if (i == 599) early = cells.size();
        }
        const std::string name = "seed " + std::to_string(seed) + ": ";
        check(s.collisions == 0, name + "no collisions in 300s");
        check(s.limitViolations == 0 && s.maxForward <= 0.4f, name + "limits respected on all " + std::to_string(s.ticks) + " ticks");
        check(cells.size() > early && cells.size() >= 30, name + "coverage grows (" + std::to_string(early) + " -> " + std::to_string(cells.size()) + " of ~88 1m cells)");
        check(forward > 20, name + "actually walked (" + num(forward) + "m)");
        check(s.reachedNotLooking == 0, name + "every arrival is followed by looking around");
    }
}

int main() {
    testNoMap();
    testStale();
    testFrameMismatch();
    testInvalidPose();
    testDeterminism();
    testGoalChoice();
    testScoring();
    testTimeoutAndFailedGoals();
    testPruning();
    testBoxedIn();
    testRegressionReached();
    testWallBetween();
    testPlanFailures();
    testTrail();
    testBlockedAhead();
    testTurnSign();
    testConfigLimits();
    testLongRun();

    std::cout << (failures == 0 ? "all checks passed" : std::to_string(failures) + " check(s) failed") << std::endl;
    return failures == 0 ? 0 : 1;
}
