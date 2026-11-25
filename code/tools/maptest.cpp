// Self-test of the ObstacleMap with synthetic LiDAR height maps instead of the robot:
// free distance, route planning and reachable spots on flat floor, walls, gaps, stairs,
// drops and a person standing at the goal, plus memory (moving the grid along,
// forgetting, old obstacles) and route hysteresis. Runs anywhere, needs no models:
//
//   cd code && ./build-tools/maptest
//
// Exit code 0 when all checks pass. "maptest --bench" also times the planner.

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include "../src/core/ObstacleMap.h"

using robo::sdk::core::HeightMapFrame;
using robo::sdk::core::ObstacleMap;
using Path = std::vector<std::pair<float, float>>;
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

// whether any route point lies in the region
static bool touches(const Path& path, const std::function<bool(float, float)>& region) {
    for (const auto& [x, y] : path)
        if (region(x, y)) return true;
    return false;
}

// the route ends at a cell center within radius of the goal's cell, so allow half a cell
static bool endsNear(const Path& path, float x, float y, float radius) {
    return !path.empty() && std::hypot(path.back().first - x, path.back().second - y) <= radius + 0.08f;
}

// which side (+1 left, -1 right) the route passes an obstacle at x
static int sideAt(const Path& path, float x) {
    float best = 1e9f, y = 0;
    for (const auto& p : path) {
        if (std::abs(p.first - x) < best) {
            best = std::abs(p.first - x);
            y = p.second;
        }
    }
    return y >= 0 ? 1 : -1;
}

static void testFlatFloor() {
    std::cout << "flat floor" << std::endl;
    TestMap t;
    check(!t.map.available(), "nothing available before the first frame");
    check(std::isinf(t.map.secondsSinceUpdate()), "secondsSinceUpdate is INFINITY before the first frame");
    t.observe(flat);
    check(t.map.available(), "available after a frame");
    check(t.map.freeDistance(0, 0, 0, 3) == 3, "free distance is the full 3m");

    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    check(ok && endsNear(path, 3, 0, 0.4f), "route to (3, 0) found and ends at the goal");
    check(ok && !touches(path, [](float, float y) { return std::abs(y) > 0.15f; }), "route is straight");

    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 1, 3, spots);
    bool inRange = !spots.empty(), frontier = false;
    for (const auto& s : spots) {
        const float d = std::hypot(s.x, s.y);
        inRange = inRange && d >= 1 && d <= 3.01f;
        frontier = frontier || s.frontier;
    }
    check(inRange, "reachable spots found, all 1-3m away (" + std::to_string(spots.size()) + ")");
    check(!frontier, "no frontier spots well inside the known area");
}

static void testWall(float res) {
    std::cout << "wall at 1.5m (local map resolution " << res << ")" << std::endl;
    TestMap t;
    const auto wall = [](float x, float y) { return inBox(x, y, 1.5f, 1.7f, -1, 1) ? 1.0f : 0.0f; };
    t.observe(wall, 0, 0, 5, res);
    const float free = t.map.freeDistance(0, 0, 0, 3);
    check(free >= 1.3f && free <= 1.6f, "free distance stops at the wall (" + std::to_string(free) + ")");
    check(t.map.freeDistance(0, 0, 3.14159265f, 3) == 3, "free behind the robot");

    Path path;
    const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
    check(ok && endsNear(path, 3, 0, 0.4f), "route around the wall found");
    check(ok && !touches(path, [](float x, float y) { return inBox(x, y, 1.3f, 1.9f, -1.2f, 1.2f); }),
          "route keeps 0.2m+ away from the wall");

    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 0.5f, 4, spots);
    bool inWall = false;
    for (const auto& s : spots) inWall = inWall || inBox(s.x, s.y, 1.5f, 1.7f, -1, 1);
    check(!spots.empty() && !inWall, "no reachable spot on the wall");
}

static void testGap() {
    std::cout << "wall with a gap" << std::endl;
    const auto crossesWall = [](float x, float y) { return x > 1.45f && x < 1.75f && std::abs(y) < 4.8f; };
    {
        TestMap t;
        // wall over the whole local map, 1.2m gap in the middle
        t.observe([](float x, float y) { return (x >= 1.5f && x <= 1.7f && std::abs(y) > 0.6f) ? 1.0f : 0.0f; });
        Path path;
        const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
        check(ok && endsNear(path, 3, 0, 0.4f), "route through a 1.2m gap found");
        check(ok && !touches(path, [](float x, float y) { return x > 1.45f && x < 1.75f && std::abs(y) > 0.3f; }),
              "route goes through the middle of the gap");
    }
    {
        TestMap t;
        // 0.4m gap: narrower than the robot
        t.observe([](float x, float y) { return (x >= 1.5f && x <= 1.7f && std::abs(y) > 0.2f) ? 1.0f : 0.0f; });
        Path path;
        const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
        check(!ok || !touches(path, crossesWall), "no route through a 0.4m gap");
        check(t.map.freeDistance(0, 0, 0, 3) < 1.6f, "free distance blocked by the narrow gap");
    }
}

static void testStairs() {
    std::cout << "stairs and drops" << std::endl;
    {
        TestMap t;
        // one 0.15m step up at 1.5m
        t.observe([](float x, float) { return x >= 1.5f ? 0.15f : 0.0f; });
        check(t.map.freeDistance(0, 0, 0, 3) == 3, "0.15m step stays free");
        Path path;
        const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
        check(ok && !touches(path, [](float, float y) { return std::abs(y) > 0.15f; }), "straight route over a 0.15m step");
    }
    {
        TestMap t;
        // staircase of 0.18m steps every 0.3m
        t.observe([](float x, float) { return x >= 1.0f ? 0.18f * (1 + std::floor((x - 1.0f) / 0.3f)) : 0.0f; });
        check(t.map.freeDistance(0, 0, 0, 3) == 3, "staircase of 0.18m steps stays free");
        Path path;
        check(t.map.planPath(0, 0, 3, 0, 0.4f, path) && !touches(path, [](float, float y) { return std::abs(y) > 0.15f; }),
              "straight route up the staircase");
    }
    {
        TestMap t;
        // 0.3m drop at 1.5m across the whole local map
        t.observe([](float x, float) { return x >= 1.5f ? -0.3f : 0.0f; });
        const float free = t.map.freeDistance(0, 0, 0, 3);
        check(free >= 1.3f && free <= 1.6f, "0.3m drop blocks (" + std::to_string(free) + ")");
        Path path;
        const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
        check(!ok || !touches(path, [](float x, float y) { return x > 1.4f && x < 1.6f && std::abs(y) < 4.8f; }),
              "no route over the drop");
    }
}

static void testPersonAtGoal() {
    std::cout << "person at the goal" << std::endl;
    const auto person = [](float cx, float cy) {
        return [cx, cy](float x, float y) { return std::hypot(x - cx, y - cy) <= 0.2f ? 1.7f : 0.0f; };
    };
    {
        TestMap t;
        t.observe(person(2, 0));
        Path path;
        const bool ok = t.map.planPath(0, 0, 2, 0, 0.4f, path);
        check(ok && endsNear(path, 2, 0, 0.4f), "route to a person standing on the goal found");
        check(ok && !touches(path, [](float x, float y) { return std::hypot(x - 2, y) < 0.3f; }),
              "route stops before the person's legs");
        const float free = t.map.freeDistance(0, 0, 0, 3);
        check(free >= 1.6f && free <= 1.9f, "free distance still stops at the person (" + std::to_string(free) + ")");
    }
    {
        TestMap t;
        // the same person in the way, goal behind them: the goal exception must not apply
        t.observe(person(1.5f, 0));
        Path path;
        const bool ok = t.map.planPath(0, 0, 3, 0, 0.4f, path);
        check(ok && !touches(path, [](float x, float y) { return std::hypot(x - 1.5f, y) < 0.5f; }),
              "person in the way is passed with full clearance");
    }
}

static void testRecenter() {
    std::cout << "moving the grid along" << std::endl;
    TestMap t;
    t.observe([](float x, float y) { return inBox(x, y, 2.0f, 2.2f, -1, 1) ? 1.0f : 0.0f; });
    t.time = 1;
    // the robot walked 11m, further than the 10m after which the grid is moved
    t.observe(flat, 11, 0, 3);
    check(t.map.contains(0, 0) && t.map.contains(11, 0), "old and new position both in the grid");
    check(!t.map.contains(-9.5f, 0), "grid moved: far side dropped");
    const float free = t.map.freeDistance(0, 0, 0, 4);
    check(free >= 1.8f && free <= 2.1f, "wall seen before the move is remembered (" + std::to_string(free) + ")");
    Path path;
    check(t.map.planPath(0, 0, 4, 0, 0.4f, path) &&
          !touches(path, [](float x, float y) { return inBox(x, y, 1.9f, 2.3f, -1.1f, 1.1f); }),
          "route still goes around the remembered wall");
}

static void testForgetting() {
    std::cout << "forgetting" << std::endl;
    TestMap t;
    t.observe([](float x, float y) { return inBox(x, y, 1.5f, 1.7f, -1, 1) ? 1.0f : 0.0f; });
    t.time = 11;
    check(t.map.freeDistance(0, 0, 0, 3) == 3, "free distance ignores obstacles older than 10s");
    check(std::abs(t.map.secondsSinceUpdate() - 11) < 1e-3f, "secondsSinceUpdate counts from the last frame");
    Path path;
    check(t.map.planPath(0, 0, 3, 0, 0.4f, path) &&
          !touches(path, [](float x, float y) { return inBox(x, y, 1.5f, 1.7f, -1, 1); }),
          "old wall still avoided by routes when going around is cheap");
    t.time = 181;
    check(t.map.planPath(0, 0, 3, 0, 0.4f, path) && !touches(path, [](float, float y) { return std::abs(y) > 0.15f; }),
          "after 3 minutes the wall is forgotten, route is straight");
    std::vector<ObstacleMap::Spot> spots;
    // (the robot's own surroundings, up to 0.5m, always count as walkable)
    t.map.reachableSpots(0, 0, 1, 3, spots);
    check(spots.empty(), "no known spots left after forgetting");
}

static void testOldObstacles() {
    std::cout << "fresh vs old obstacles (closed room)" << std::endl;
    TestMap t;
    // the robot stands in a closed 3x3m room
    t.observe([](float x, float y) {
        const float m = std::max(std::abs(x), std::abs(y));
        return (m >= 1.5f && m <= 1.7f) ? 1.0f : 0.0f;
    });
    Path path;
    check(!t.map.planPath(0, 0, 3, 0, 0.4f, path), "fresh walls: no way out");
    std::vector<ObstacleMap::Spot> spots;
    t.map.reachableSpots(0, 0, 0.5f, 5, spots);
    bool inside = !spots.empty();
    for (const auto& s : spots) inside = inside && std::max(std::abs(s.x), std::abs(s.y)) < 1.5f;
    check(inside, "reachable spots only inside the room");

    t.time = 20;
    check(t.map.planPath(0, 0, 3, 0, 0.4f, path) && endsNear(path, 3, 0, 0.4f),
          "old walls (may have been people) only cost extra: route found");
    t.map.reachableSpots(0, 0, 0.5f, 5, spots);
    inside = !spots.empty();
    for (const auto& s : spots) inside = inside && std::max(std::abs(s.x), std::abs(s.y)) < 1.5f;
    check(inside, "goals are still not picked behind old walls");
}

static void testHysteresis() {
    std::cout << "route hysteresis" << std::endl;
    const auto post = [](float x, float y) { return std::hypot(x - 2, y) <= 0.4f ? 1.0f : 0.0f; };
    Path path;
    {
        // without memory the slightly shorter side wins
        TestMap fresh;
        fresh.observe(post);
        fresh.map.planPath(0, 0, 4, -0.2f, 0.4f, path);
        const int right = sideAt(path, 2);
        TestMap other;
        other.observe(post);
        other.map.planPath(0, 0, 4, 0.2f, 0.4f, path);
        check(right == -1 && sideAt(path, 2) == 1, "without a previous route the shorter side is taken");
    }

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
    check(stable, "symmetric post: the route keeps its side over 20 jittering plans");

    // the remembered side gets blocked: the route must switch
    t.time += 0.1f;
    t.observe([&](float x, float y) {
        if (post(x, y) > 0) return 1.0f;
        return (x >= 1.9f && x <= 2.1f && y * side > 0) ? 1.0f : 0.0f;
    });
    check(t.map.planPath(0, 0, 4, 0, 0.4f, path) && sideAt(path, 2) == -side,
          "route switches sides when the remembered side gets blocked");
}

static void testGroundBaseline() {
    std::cout << "obstacle starting under the robot's nose" << std::endl;
    {
        TestMap t;
        // a 0.4m box from 0.2m to 1.0m ahead: the rays start inside it at 0.3m
        t.observe([](float x, float y) { return inBox(x, y, 0.2f, 1.0f, -0.4f, 0.4f) ? 0.4f : 0.0f; });
        const float free = t.map.freeDistance(0, 0, 0, 3);
        check(free <= 0.35f, "box that starts closer than 0.3m blocks right away (" + std::to_string(free) + ")");
    }
    {
        TestMap t;
        // a stair step starting 0.2m ahead must still be free
        t.observe([](float x, float) { return x >= 0.2f ? 0.15f : 0.0f; });
        check(t.map.freeDistance(0, 0, 0, 3) == 3, "0.15m step starting under the nose stays free");
    }
}

static void testFrame() {
    std::cout << "frame sanity" << std::endl;
    TestMap t;
    check(t.map.checkFrame(5, 5), "no frame yet: nothing to complain about");
    t.time = 3;
    t.observe(flat, 1, 1);
    t.time = 3.5f;
    check(std::abs(t.map.secondsSinceUpdate() - 0.5f) < 1e-3f, "secondsSinceUpdate = 0.5s");
    check(t.map.checkFrame(1.5f, 1), "map centered on the robot: consistent");
    check(!t.map.checkFrame(5, 1), "map 4m away from the robot: mismatch reported");

    // unobserved and broken cells don't overwrite what is known
    t.observe([](float x, float y) { return inBox(x, y, 1.5f, 1.7f, -1, 3) ? 1.0f : 0.0f; }, 1, 1);
    t.observe([](float, float) { return NAN; }, 1, 1);
    const float free = t.map.freeDistance(1, 1, 0, 3);
    check(free <= 0.75f, "unobserved cells keep the remembered wall (" + std::to_string(free) + ")");
}

// times planning in a follow-like scene: walls, clutter and a person at the goal
static void bench() {
    std::cout << "timing" << std::endl;
    TestMap t;
    const auto scene = [](float x, float y) {
        if (std::hypot(x - 3, y) <= 0.2f) return 1.7f;                          // the person
        if (inBox(x, y, 1.5f, 1.7f, -1.5f, 0.8f)) return 1.0f;                  // a wall in the way
        // clutter: 0.3m boxes every 1.6m, away from the robot and the person
        const float cx = x - 1.6f * std::round(x / 1.6f), cy = y - 1.6f * std::round(y / 1.6f) - 0.8f;
        if (std::abs(cx) < 0.15f && std::abs(cy) < 0.15f && std::hypot(x, y) > 1 && std::hypot(x - 3, y) > 1) return 0.5f;
        return 0.0f;
    };
    t.observe(scene, 0, 0, 4);
    Path path;
    std::vector<ObstacleMap::Spot> spots;
    const int runs = 200;
    auto start = std::chrono::steady_clock::now();
    int found = 0;
    // the clock stays put so every obstacle stays fresh
    for (int i = 0; i < runs; i++) found += t.map.planPath(0, 0, 3, 0, 0.4f, path);
    const float planMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count() / runs;
    start = std::chrono::steady_clock::now();
    int foundFree = 0;
    for (int i = 0; i < runs; i++) foundFree += t.map.planPath(0, 0, 2.5f, 1.6f, 0.4f, path);
    const float planFreeMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count() / runs;
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; i++) t.map.reachableSpots(0, 0, 1.5f, 8, spots);
    const float spotsMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count() / runs;
    start = std::chrono::steady_clock::now();
    float sum = 0;
    for (int i = 0; i < runs; i++) sum += t.map.freeDistance(0, 0, 0.1f * (i % 10), 3);
    const float freeUs = std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - start).count() / runs;
    std::cout << "  planPath (3m, person at goal): " << planMs << " ms, found " << found << "/" << runs << std::endl;
    std::cout << "  planPath (3m, free goal): " << planFreeMs << " ms, found " << foundFree << "/" << runs << std::endl;
    std::cout << "  reachableSpots (8m): " << spotsMs << " ms, " << spots.size() << " spots" << std::endl;
    std::cout << "  freeDistance (3m): " << freeUs << " us (sum " << sum << ")" << std::endl;
}

int main(int argc, char** argv) {
    testFlatFloor();
    testWall(0.1f);
    testWall(0.06f);
    testGap();
    testStairs();
    testPersonAtGoal();
    testRecenter();
    testForgetting();
    testOldObstacles();
    testHysteresis();
    testGroundBaseline();
    testFrame();
    if (argc > 1 && std::string(argv[1]) == "--bench") bench();

    std::cout << (failures == 0 ? "all checks passed" : std::to_string(failures) + " check(s) failed") << std::endl;
    return failures == 0 ? 0 : 1;
}
