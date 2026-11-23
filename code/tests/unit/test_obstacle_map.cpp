// ObstacleMap: remembered LiDAR terrain, free distances, route planning and reachable spots
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include "doctest/doctest.h"
#include "TestMap.h"

using namespace testutil;

namespace {

// a hand-built frame whose data may be shorter than width * height or hold broken values,
// which TestMap::observe can't produce
struct RawFrame {
    std::vector<float> data;
    HeightMapFrame frame;
};

// raw: terrain value written as is (no NAN conversion). dataCells < 0 means all cells
RawFrame makeRaw(const Terrain& raw, float half = 5.0f, int dataCells = -1, float cx = 0, float cy = 0) {
    const float res = 0.1f;
    const int n = static_cast<int>(std::lround(2 * half / res));
    RawFrame r;
    r.frame.resolution = res;
    r.frame.width = r.frame.height = n;
    r.frame.originX = cx - n * res / 2;
    r.frame.originY = cy - n * res / 2;
    const size_t total = dataCells < 0 ? static_cast<size_t>(n) * n : static_cast<size_t>(dataCells);
    // exactly as many floats as announced, so any read past the end is visible to sanitizers
    r.data.resize(total);
    for (size_t i = 0; i < total; i++) {
        const int x = static_cast<int>(i % n), y = static_cast<int>(i / n);
        r.data[i] = raw(r.frame.originX + (x + 0.5f) * res, r.frame.originY + (y + 0.5f) * res);
    }
    r.frame.data = r.data.data();
    r.frame.size = total;
    return r;
}

float wallTerrain(float x, float y) { return inBox(x, y, 1.5f, 1.7f, -1, 1) ? 1.0f : 0.0f; }

bool straight(const Path& path) {
    return !touches(path, [](float, float y) { return std::abs(y) > 0.15f; });
}

// captures std::cerr while alive
struct CerrCapture {
    std::ostringstream buffer;
    std::streambuf* old;
    CerrCapture() : old(std::cerr.rdbuf(buffer.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(old); }
    CerrCapture(const CerrCapture&) = delete;
    CerrCapture& operator=(const CerrCapture&) = delete;
};

// maptest: testWall
void checkWallFreeDistance(float res) {
    TestMap t;
    t.observe(wallTerrain, 0, 0, 5, res);
    const float free = t.map.freeDistance(0, 0, 0, 3);
    CAPTURE(free);
    CHECK(free >= 1.3f);
    CHECK(free <= 1.6f);
    CHECK(t.map.freeDistance(0, 0, 3.14159265f, 3) == 3);
}

void checkWallRoute(float res) {
    TestMap t;
    t.observe(wallTerrain, 0, 0, 5, res);
    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    REQUIRE(ok);
    CHECK(endsNear(path, 3, 0, 0.4f));
    CHECK_FALSE(touches(path, [](float x, float y) { return inBox(x, y, 1.3f, 1.9f, -1.2f, 1.2f); }));

    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 0.5f, 4, spots);
    bool inWall = false;
    for (const auto& s : spots) inWall = inWall || inBox(s.x, s.y, 1.5f, 1.7f, -1, 1);
    CHECK_FALSE(spots.empty());
    CHECK_FALSE(inWall);
}

}

TEST_CASE("nothing is available before the first frame") {
    TestMap t;
    CHECK_FALSE(t.map.available());
    t.observe(flat);
    CHECK(t.map.available());
}

// maptest: testFlatFloor (before the first frame)
TEST_CASE("an empty map reports no update and no known area") {
    TestMap t;
    CHECK_FALSE(t.map.available());
    CHECK(std::isinf(t.map.secondsSinceUpdate()));
    CHECK_FALSE(t.map.contains(0, 0));
}

TEST_CASE("the known area ends at the 40 m grid") {
    TestMap t;
    t.observe(flat);
    CHECK(t.map.contains(0, 0));
    CHECK(t.map.contains(15, -15));
    CHECK_FALSE(t.map.contains(25, 0));
    CHECK_FALSE(t.map.contains(0, -25));
}

// maptest: testFlatFloor
TEST_CASE("flat floor is free and the route to a goal is straight") {
    TestMap t;
    t.observe(flat);
    CHECK(t.map.freeDistance(0, 0, 0, 3) == 3);

    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    REQUIRE(ok);
    CHECK(endsNear(path, 3, 0, 0.4f));
    CHECK(straight(path));
}

// maptest: testFlatFloor
TEST_CASE("flat floor spots are all in range and none are frontier") {
    TestMap t;
    t.observe(flat);
    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 1, 3, spots);
    bool inRange = !spots.empty(), frontier = false;
    for (const auto& s : spots) {
        const float d = std::hypot(s.x, s.y);
        inRange = inRange && d >= 1 && d <= 3.01f;
        frontier = frontier || s.frontier;
    }
    CHECK(inRange);
    CHECK_FALSE(frontier);
}

// maptest: testWall (resolution 0.1)
TEST_CASE("a wall at 1.5 m stops the free distance with a 0.1 m local map") {
    checkWallFreeDistance(0.1f);
}

// maptest: testWall (resolution 0.1)
TEST_CASE("the route and spots avoid a wall at 1.5 m with a 0.1 m local map") {
    checkWallRoute(0.1f);
}

// maptest: testWall (resolution 0.06)
TEST_CASE("a wall at 1.5 m stops the free distance with a 0.06 m local map") {
    checkWallFreeDistance(0.06f);
}

// maptest: testWall (resolution 0.06)
TEST_CASE("the route and spots avoid a wall at 1.5 m with a 0.06 m local map") {
    checkWallRoute(0.06f);
}

// maptest: testGap
TEST_CASE("a route goes through the middle of a 1.2 m gap") {
    TestMap t;
    // wall over the whole local map, 1.2m gap in the middle
    t.observe([](float x, float y) { return (x >= 1.5f && x <= 1.7f && std::abs(y) > 0.6f) ? 1.0f : 0.0f; });
    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    REQUIRE(ok);
    CHECK(endsNear(path, 3, 0, 0.4f));
    CHECK_FALSE(touches(path, [](float x, float y) { return x > 1.45f && x < 1.75f && std::abs(y) > 0.3f; }));
}

// maptest: testGap
TEST_CASE("no route and no free distance through a gap narrower than the robot") {
    const auto crossesWall = [](float x, float y) { return x > 1.45f && x < 1.75f && std::abs(y) < 4.8f; };
    TestMap t;
    // 0.4m gap: narrower than the robot
    t.observe([](float x, float y) { return (x >= 1.5f && x <= 1.7f && std::abs(y) > 0.2f) ? 1.0f : 0.0f; });
    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    CHECK((!ok || !touches(path, crossesWall)));
    CHECK(t.map.freeDistance(0, 0, 0, 3) < 1.6f);
}

// maptest: testStairs
TEST_CASE("a single 0.15 m step stays free and the route stays straight") {
    TestMap t;
    // one 0.15m step up at 1.5m
    t.observe([](float x, float) { return x >= 1.5f ? 0.15f : 0.0f; });
    CHECK(t.map.freeDistance(0, 0, 0, 3) == 3);
    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    REQUIRE(ok);
    CHECK(straight(path));
}

// maptest: testStairs
TEST_CASE("a staircase of 0.18 m steps stays free and the route stays straight") {
    TestMap t;
    // staircase of 0.18m steps every 0.3m
    t.observe([](float x, float) { return x >= 1.0f ? 0.18f * (1 + std::floor((x - 1.0f) / 0.3f)) : 0.0f; });
    CHECK(t.map.freeDistance(0, 0, 0, 3) == 3);
    Path path;
    REQUIRE(t.map.planPath(0, 0, 3, 0, 0.4f, path));
    CHECK(straight(path));
}

// maptest: testStairs
TEST_CASE("a 0.3 m drop blocks the free distance and the route") {
    TestMap t;
    // 0.3m drop at 1.5m across the whole local map
    t.observe([](float x, float) { return x >= 1.5f ? -0.3f : 0.0f; });
    const float free = t.map.freeDistance(0, 0, 0, 3);
    CAPTURE(free);
    CHECK(free >= 1.3f);
    CHECK(free <= 1.6f);
    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    CHECK((!ok || !touches(path, [](float x, float y) { return x > 1.4f && x < 1.6f && std::abs(y) < 4.8f; })));
}

// maptest: testPersonAtGoal
TEST_CASE("a route to a person standing on the goal stops before the legs") {
    TestMap t;
    t.observe([](float x, float y) { return std::hypot(x - 2, y) <= 0.2f ? 1.7f : 0.0f; });
    Path path;
    const bool ok = t.map.planPath(0, 0, 2, 0, 0.4f, path);
    REQUIRE(ok);
    CHECK(endsNear(path, 2, 0, 0.4f));
    CHECK_FALSE(touches(path, [](float x, float y) { return std::hypot(x - 2, y) < 0.3f; }));
    const float free = t.map.freeDistance(0, 0, 0, 3);
    CAPTURE(free);
    CHECK(free >= 1.6f);
    CHECK(free <= 1.9f);
}

// maptest: testPersonAtGoal
TEST_CASE("a person in the way is passed with full clearance") {
    TestMap t;
    // the person is in the way, the goal is behind them: the goal exception must not apply
    t.observe([](float x, float y) { return std::hypot(x - 1.5f, y) <= 0.2f ? 1.7f : 0.0f; });
    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    REQUIRE(ok);
    CHECK_FALSE(touches(path, [](float x, float y) { return std::hypot(x - 1.5f, y) < 0.5f; }));
}

// maptest: testRecenter
TEST_CASE("moving the grid along keeps old and new positions inside and drops the far side") {
    TestMap t;
    t.observe(wallTerrain);
    t.time = 1;
    // the robot walked 11m, further than the 10m after which the grid is moved
    t.observe(flat, 11, 0, 3);
    CHECK(t.map.contains(0, 0));
    CHECK(t.map.contains(11, 0));
    CHECK_FALSE(t.map.contains(-9.5f, 0));
}

// maptest: testRecenter
TEST_CASE("a wall seen before the grid moved is still remembered and avoided") {
    TestMap t;
    t.observe([](float x, float y) { return inBox(x, y, 2.0f, 2.2f, -1, 1) ? 1.0f : 0.0f; });
    t.time = 1;
    t.observe(flat, 11, 0, 3);
    const float free = t.map.freeDistance(0, 0, 0, 4);
    CAPTURE(free);
    CHECK(free >= 1.8f);
    CHECK(free <= 2.1f);
    Path path;
    REQUIRE(t.map.planPath(0, 0, 4, 0, 0.4f, path));
    CHECK_FALSE(touches(path, [](float x, float y) { return inBox(x, y, 1.9f, 2.3f, -1.1f, 1.1f); }));
}

// maptest: testForgetting
TEST_CASE("free distance ignores obstacles older than 10 s but the route still avoids them") {
    TestMap t;
    t.observe(wallTerrain);
    t.time = 11;
    CHECK(t.map.freeDistance(0, 0, 0, 3) == 3);
    CHECK(std::abs(t.map.secondsSinceUpdate() - 11) < 1e-3f);
    Path path;
    REQUIRE(t.map.planPath(0, 0, 3, 0, 0.4f, path));
    // old wall still avoided by routes when going around is cheap
    CHECK_FALSE(touches(path, [](float x, float y) { return inBox(x, y, 1.5f, 1.7f, -1, 1); }));
}

// maptest: testForgetting
TEST_CASE("after 3 minutes the wall is forgotten and no spots are left") {
    TestMap t;
    t.observe(wallTerrain);
    t.time = 181;
    Path path;
    REQUIRE(t.map.planPath(0, 0, 3, 0, 0.4f, path));
    CHECK(straight(path));
    std::vector<ObstacleMap::Spot> spots;
    // (the robot's own surroundings, up to 0.5m, always count as walkable)
    t.map.reachableSpots(0, 0, 1, 3, spots);
    CHECK(spots.empty());
}

// maptest: testOldObstacles
TEST_CASE("fresh walls of a closed room leave no way out and spots only inside") {
    TestMap t;
    // the robot stands in a closed 3x3m room
    t.observe([](float x, float y) {
        const float m = std::max(std::abs(x), std::abs(y));
        return (m >= 1.5f && m <= 1.7f) ? 1.0f : 0.0f;
    });
    Path path;
    CHECK_FALSE(t.map.planPath(0, 0, 3, 0, 0.4f, path));
    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 0.5f, 5, spots);
    bool inside = !spots.empty();
    for (const auto& s : spots) inside = inside && std::max(std::abs(s.x), std::abs(s.y)) < 1.5f;
    CHECK(inside);
}

// maptest: testOldObstacles
TEST_CASE("old walls of a closed room only cost extra and goals stay inside") {
    TestMap t;
    t.observe([](float x, float y) {
        const float m = std::max(std::abs(x), std::abs(y));
        return (m >= 1.5f && m <= 1.7f) ? 1.0f : 0.0f;
    });
    t.time = 20;
    Path path;
    // old walls may have been people
    REQUIRE(t.map.planPath(0, 0, 3, 0, 0.4f, path));
    CHECK(endsNear(path, 3, 0, 0.4f));
    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 0.5f, 5, spots);
    bool inside = !spots.empty();
    for (const auto& s : spots) inside = inside && std::max(std::abs(s.x), std::abs(s.y)) < 1.5f;
    CHECK(inside);
}

// maptest: testHysteresis
TEST_CASE("without a previous route the shorter side of a post is taken") {
    const auto post = [](float x, float y) { return std::hypot(x - 2, y) <= 0.4f ? 1.0f : 0.0f; };
    Path path;
    TestMap fresh;
    fresh.observe(post);
    fresh.map.planPath(0, 0, 4, -0.2f, 0.4f, path);
    const int right = sideAt(path, 2);
    TestMap other;
    other.observe(post);
    other.map.planPath(0, 0, 4, 0.2f, 0.4f, path);
    CHECK(right == -1);
    CHECK(sideAt(path, 2) == 1);
}

// maptest: testHysteresis
TEST_CASE("a route about a symmetric post keeps its side while the goal jitters") {
    const auto post = [](float x, float y) { return std::hypot(x - 2, y) <= 0.4f ? 1.0f : 0.0f; };
    Path path;
    TestMap t;
    t.observe(post);
    t.map.planPath(0, 0, 4, 0.2f, 0.4f, path);
    const int side = sideAt(path, 2);
    bool stable = true;
    for (int i = 0; i < 20; i++) {
        t.time += 0.1f;
        // the goal and the robot jitter around the symmetric line
        const float jitter = (i % 2 ? 1.0f : -1.0f) * 0.2f;
        stable = stable && t.map.planPath(0, -jitter / 4, 4, jitter, 0.4f, path) && sideAt(path, 2) == side;
    }
    CHECK(stable);
}

// maptest: testHysteresis
TEST_CASE("the route switches sides when the remembered side gets blocked") {
    const auto post = [](float x, float y) { return std::hypot(x - 2, y) <= 0.4f ? 1.0f : 0.0f; };
    Path path;
    TestMap t;
    t.observe(post);
    t.map.planPath(0, 0, 4, 0.2f, 0.4f, path);
    const int side = sideAt(path, 2);
    for (int i = 0; i < 20; i++) {
        t.time += 0.1f;
        const float jitter = (i % 2 ? 1.0f : -1.0f) * 0.2f;
        t.map.planPath(0, -jitter / 4, 4, jitter, 0.4f, path);
    }

    // the remembered side gets blocked: the route must switch
    t.time += 0.1f;
    t.observe([&](float x, float y) {
        if (post(x, y) > 0) return 1.0f;
        return (x >= 1.9f && x <= 2.1f && y * side > 0) ? 1.0f : 0.0f;
    });
    REQUIRE(t.map.planPath(0, 0, 4, 0, 0.4f, path));
    CHECK(sideAt(path, 2) == -side);
}

// maptest: testGroundBaseline
TEST_CASE("a box starting closer than 0.3 m blocks right away") {
    TestMap t;
    // a 0.4m box from 0.2m to 1.0m ahead: the rays start inside it at 0.3m
    t.observe([](float x, float y) { return inBox(x, y, 0.2f, 1.0f, -0.4f, 0.4f) ? 0.4f : 0.0f; });
    const float free = t.map.freeDistance(0, 0, 0, 3);
    CAPTURE(free);
    CHECK(free <= 0.35f);
}

// maptest: testGroundBaseline
TEST_CASE("a 0.15 m step starting under the nose stays free") {
    TestMap t;
    // a stair step starting 0.2m ahead must still be free
    t.observe([](float x, float) { return x >= 0.2f ? 0.15f : 0.0f; });
    CHECK(t.map.freeDistance(0, 0, 0, 3) == 3);
}

// maptest: testFrame
TEST_CASE("the frame check and the update age follow the last frame") {
    TestMap t;
    CHECK(t.map.checkFrame(5, 5));
    t.time = 3;
    t.observe(flat, 1, 1);
    t.time = 3.5f;
    CHECK(std::abs(t.map.secondsSinceUpdate() - 0.5f) < 1e-3f);
    CHECK(t.map.checkFrame(1.5f, 1));
    CHECK_FALSE(t.map.checkFrame(5, 1));
}

// maptest: testFrame
TEST_CASE("unobserved cells keep the remembered wall") {
    TestMap t;
    t.time = 3;
    t.observe(flat, 1, 1);
    t.observe([](float x, float y) { return inBox(x, y, 1.5f, 1.7f, -1, 3) ? 1.0f : 0.0f; }, 1, 1);
    t.observe([](float, float) { return NAN; }, 1, 1);
    const float free = t.map.freeDistance(1, 1, 0, 3);
    CAPTURE(free);
    CHECK(free <= 0.75f);
}

TEST_CASE("frames with a bad resolution or size or no data are ignored") {
    TestMap t;
    const std::vector<float> data(100, 0.0f);
    HeightMapFrame good;
    good.resolution = 0.1f;
    good.width = good.height = 10;
    good.originX = good.originY = -0.5f;
    good.data = data.data();
    good.size = data.size();

    HeightMapFrame f = good;
    f.resolution = 0;
    t.map.integrate(f);
    CHECK_FALSE(t.map.available());
    f = good;
    f.resolution = -0.1f;
    t.map.integrate(f);
    CHECK_FALSE(t.map.available());
    f = good;
    f.width = 0;
    t.map.integrate(f);
    CHECK_FALSE(t.map.available());
    f = good;
    f.height = -3;
    t.map.integrate(f);
    CHECK_FALSE(t.map.available());
    f = good;
    f.data = nullptr;
    t.map.integrate(f);
    CHECK_FALSE(t.map.available());
    CHECK(std::isinf(t.map.secondsSinceUpdate()));

    // the same frame with valid fields is accepted
    t.map.integrate(good);
    CHECK(t.map.available());
}

TEST_CASE("a frame whose data is shorter than width times height only writes the covered cells") {
    TestMap t;
    // the first half of the rows (y < 0) is delivered, the wall in the other half is not
    const int n = 100;
    RawFrame r = makeRaw([](float x, float y) {
        return (inBox(x, y, 1.5f, 1.7f, 1, 2) || inBox(x, y, 1.5f, 1.7f, -2, -1)) ? 1.0f : 0.0f;
    }, 5.0f, n * n / 2);
    t.map.integrate(r.frame);
    REQUIRE(t.map.available());

    const float seen = t.map.freeDistance(0, -1.5f, 0, 3);
    CAPTURE(seen);
    CHECK(seen < 1.6f);
    // nothing was delivered up there: no wall, and no known ground either
    CHECK(t.map.freeDistance(0, 1.5f, 0, 3) == 3);
}

TEST_CASE("a frame with size zero writes nothing") {
    TestMap t;
    RawFrame r = makeRaw(wallTerrain, 5.0f, 0);
    // no cells covered at all (data pointer stays non-null but size is zero)
    std::vector<float> dummy(1, 0.0f);
    r.frame.data = dummy.data();
    r.frame.size = 0;
    t.map.integrate(r.frame);
    CHECK(t.map.available());
    CHECK(t.map.freeDistance(0, 0, 0, 3) == 3);
    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 1, 3, spots);
    // only the robot's own surroundings count as walkable, nothing is known beyond them
    CHECK(spots.empty());
}

TEST_CASE("NaN and infinite and huge heights do not overwrite remembered terrain") {
    TestMap t;
    t.observe(wallTerrain);
    const float before = t.map.freeDistance(0, 0, 0, 3);
    REQUIRE(before < 1.6f);

    const std::vector<float> broken = {NAN, INFINITY, -INFINITY, 100.0f, -100.0f, 1e9f, -1e9f, 150.0f};
    for (const float value : broken) {
        CAPTURE(value);
        RawFrame r = makeRaw([value](float, float) { return value; });
        t.map.integrate(r.frame);
        CHECK(t.map.freeDistance(0, 0, 0, 3) == before);
    }
}

TEST_CASE("planning before any frame fails with an empty path") {
    TestMap t;
    Path path = {{1.0f, 2.0f}};
    CHECK_FALSE(t.map.planPath(0, 0, 3, 0, 0.4f, path));
    CHECK(path.empty());
    std::vector<ObstacleMap::Spot> spots = {{1.0f, 1.0f, false}};
    t.map.reachableSpots(0, 0, 0, 3, spots);
    CHECK(spots.empty());
    // without a map there is nothing to stop the robot
    CHECK(t.map.freeDistance(0, 0, 0, 3) == 3);
}

TEST_CASE("a start or goal outside the grid cannot be planned") {
    TestMap t;
    t.observe(flat);
    Path path = {{1.0f, 2.0f}};
    CHECK_FALSE(t.map.planPath(100, 0, 3, 0, 0.4f, path));
    CHECK(path.empty());
    CHECK_FALSE(t.map.planPath(0, 0, 0, 100, 0.4f, path));
    CHECK(path.empty());
    CHECK_FALSE(t.map.planPath(-25, 0, 3, 0, 0.4f, path));
    CHECK_FALSE(t.map.planPath(0, 0, 3, -25, 0.4f, path));
    std::vector<ObstacleMap::Spot> spots = {{1.0f, 1.0f, false}};
    t.map.reachableSpots(100, 0, 0, 3, spots);
    CHECK(spots.empty());
}

TEST_CASE("a route starts at the start cell centre and moves one neighbouring cell at a time") {
    TestMap t;
    // a wall makes the route bend so diagonal and straight steps both occur
    t.observe(wallTerrain);
    Path path;
    REQUIRE(t.map.planPath(0, 0, 3, 0, 0.4f, path));
    REQUIRE(path.size() > 2);

    // the grid was centered on (0, 0): cell borders lie at multiples of 0.1 from -20
    const auto centreOffset = [](float v) {
        const float cells = (v + 20.0f) / 0.1f - 0.5f;
        return std::abs(cells - std::round(cells));
    };
    CHECK(std::hypot(path[0].first - 0, path[0].second - 0) <= 0.05f * std::sqrt(2.0f) + 1e-3f);
    CHECK(centreOffset(path[0].first) < 0.02f);
    CHECK(centreOffset(path[0].second) < 0.02f);

    const float maxStep = 0.1f * std::sqrt(2.0f) + 1e-4f;
    bool neighbours = true;
    for (size_t i = 1; i < path.size(); i++) {
        const float d = std::hypot(path[i].first - path[i - 1].first, path[i].second - path[i - 1].second);
        neighbours = neighbours && d > 0.05f && d <= maxStep;
    }
    CHECK(neighbours);
}

TEST_CASE("a goal that is already reached gives a one point route") {
    TestMap t;
    t.observe(flat);
    Path path;
    REQUIRE(t.map.planPath(0, 0, 0.1f, 0, 0.4f, path));
    CHECK(path.size() == 1);
}

TEST_CASE("secondsSinceUpdate grows with the fake clock") {
    TestMap t;
    t.time = 5;
    t.observe(flat);
    CHECK(t.map.secondsSinceUpdate() == doctest::Approx(0.0f).epsilon(1e-3));
    t.time = 12;
    CHECK(std::abs(t.map.secondsSinceUpdate() - 7) < 1e-3f);
    t.observe(flat);
    CHECK(std::abs(t.map.secondsSinceUpdate()) < 1e-3f);
}

TEST_CASE("the frame mismatch warning is printed only once") {
    TestMap t;
    t.observe(flat);
    CerrCapture capture;
    // consistent frames print nothing
    CHECK(t.map.checkFrame(0.5f, 0));
    CHECK(capture.buffer.str().empty());

    CHECK_FALSE(t.map.checkFrame(10, 0));
    const std::string first = capture.buffer.str();
    CHECK_FALSE(first.empty());
    CHECK(first.find("ObstacleMap") != std::string::npos);

    CHECK_FALSE(t.map.checkFrame(-10, 3));
    CHECK(capture.buffer.str() == first);
}

TEST_CASE("reachable spots lie between the minimum and the maximum distance") {
    TestMap t;
    t.observe(flat);
    std::vector<ObstacleMap::Spot> spots;
    const float minD = 1.0f, maxD = 3.0f;
    t.map.reachableSpots(0, 0, minD, maxD, spots);
    REQUIRE_FALSE(spots.empty());
    for (const auto& s : spots) {
        const float d = std::hypot(s.x, s.y);
        CAPTURE(s.x);
        CAPTURE(s.y);
        CHECK(d >= minD - 0.08f);
        CHECK(d <= maxD + 0.08f);
    }
}

TEST_CASE("reachable spots sit on a 0.5 m lattice") {
    TestMap t;
    t.observe(flat);
    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 0.5f, 3, spots);
    REQUIRE(spots.size() > 4);
    const auto multipleOfHalf = [](float diff) {
        const float k = diff / 0.5f;
        return std::abs(k - std::round(k)) * 0.5f < 1e-3f;
    };
    bool lattice = true;
    for (const auto& s : spots) {
        lattice = lattice && multipleOfHalf(s.x - spots[0].x) && multipleOfHalf(s.y - spots[0].y);
    }
    CHECK(lattice);
}

TEST_CASE("spots next to unseen terrain are frontier and inner spots are not") {
    TestMap t;
    // a small frame: its edge is the edge of the known map
    t.observe(flat, 0, 0, 1.5f);
    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 0.5f, 3, spots);
    REQUIRE_FALSE(spots.empty());

    bool anyFrontier = false, anyInner = false, frontierNearEdge = true, innerInside = true;
    for (const auto& s : spots) {
        const float m = std::max(std::abs(s.x), std::abs(s.y));
        if (s.frontier) {
            anyFrontier = true;
            frontierNearEdge = frontierNearEdge && m >= 1.3f;
        } else {
            anyInner = true;
            innerInside = innerInside && m < 1.3f;
        }
    }
    CHECK(anyFrontier);
    CHECK(anyInner);
    CHECK(frontierNearEdge);
    CHECK(innerInside);
}

TEST_CASE("an unreachable goal in a closed room fails quickly") {
    TestMap t;
    t.observe([](float x, float y) {
        const float m = std::max(std::abs(x), std::abs(y));
        return (m >= 1.5f && m <= 1.7f) ? 1.0f : 0.0f;
    });
    Path path;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = t.map.planPath(0, 0, 8, 0, 0.4f, path);
    const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    CHECK_FALSE(ok);
    CHECK(path.empty());
    CAPTURE(seconds);
    CHECK(seconds < 1.0f);
}

TEST_CASE("a remembered wall keeps its world position after the grid moved") {
    TestMap t;
    t.observe(wallTerrain);
    t.time = 1;
    // two recenterings in a row, the second frame does not cover the wall
    t.observe(flat, 11, 0, 3);
    t.time = 2;
    t.observe(flat, 0, 6, 1.5f);

    // seen from the other side the wall's face is at x = 1.7
    const float free = t.map.freeDistance(3.0f, 0, 3.14159265f, 3);
    CAPTURE(free);
    CHECK(free >= 1.0f);
    CHECK(free <= 1.6f);
    // and from the front its face is at x = 1.5
    const float front = t.map.freeDistance(0, 0, 0, 3);
    CAPTURE(front);
    CHECK(front >= 1.3f);
    CHECK(front <= 1.6f);
    CHECK(t.map.contains(1.6f, 0));
}

TEST_CASE("planning and queries stay sane while another thread integrates frames") {
    TestMap t;
    // the clock stays put: the lambda reads t.time from both threads
    t.observe(flat);
    std::atomic<bool> stop{false};
    std::atomic<int> frames{0};
    std::thread writer([&] {
        while (!stop.load()) {
            t.observe(flat);
            frames++;
        }
    });

    bool plansOk = true, freeOk = true, spotsOk = true;
    int rounds = 0;
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    Path path;
    std::vector<ObstacleMap::Spot> spots;
    while (std::chrono::steady_clock::now() < end) {
        plansOk = plansOk && t.map.planPath(0, 0, 3, 0, 0.4f, path) && endsNear(path, 3, 0, 0.4f);
        freeOk = freeOk && t.map.freeDistance(0, 0, 0, 3) == 3;
        t.map.reachableSpots(0, 0, 1, 3, spots);
        spotsOk = spotsOk && !spots.empty();
        rounds++;
    }
    stop = true;
    writer.join();

    CHECK(rounds > 0);
    CHECK(frames.load() > 0);
    CHECK(plansOk);
    CHECK(freeOk);
    CHECK(spotsOk);
}
