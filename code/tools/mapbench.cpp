// Times the ObstacleMap's route planning in a follow-like scene (walls, clutter and a
// person at the goal), with synthetic LiDAR height maps. Needs neither the robot nor the
// models. Run it on the Jetson to see what planning costs there:
//
//   ./build/mapbench
//
// The checks of the map itself are tests now: tests/unit/test_obstacle_map.cpp.

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include "../src/core/ObstacleMap.h"

using robo::sdk::core::HeightMapFrame;
using robo::sdk::core::ObstacleMap;
using Path = std::vector<std::pair<float, float>>;

// one LiDAR frame of the terrain, 2 * half meters wide around (0, 0)
static void observe(ObstacleMap& map, const std::function<float(float, float)>& terrain, float half, float res = 0.1f) {
    const int n = static_cast<int>(std::lround(2 * half / res));
    std::vector<float> data(static_cast<size_t>(n) * n);
    HeightMapFrame frame;
    frame.resolution = res;
    frame.width = frame.height = n;
    frame.originX = -n * res / 2;
    frame.originY = -n * res / 2;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            data[static_cast<size_t>(y) * n + x] = terrain(frame.originX + (x + 0.5f) * res, frame.originY + (y + 0.5f) * res);
    frame.data = data.data();
    frame.size = data.size();
    map.integrate(frame);
}

static bool inBox(float x, float y, float x0, float x1, float y0, float y1) {
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

int main() {
    ObstacleMap map;
    // the clock stays put so every obstacle stays fresh
    map.setClock([] { return 0.0f; });
    observe(map, [](float x, float y) {
        if (std::hypot(x - 3, y) <= 0.2f) return 1.7f;                          // the person
        if (inBox(x, y, 1.5f, 1.7f, -1.5f, 0.8f)) return 1.0f;                  // a wall in the way
        // clutter: 0.3m boxes every 1.6m, away from the robot and the person
        const float cx = x - 1.6f * std::round(x / 1.6f), cy = y - 1.6f * std::round(y / 1.6f) - 0.8f;
        if (std::abs(cx) < 0.15f && std::abs(cy) < 0.15f && std::hypot(x, y) > 1 && std::hypot(x - 3, y) > 1) return 0.5f;
        return 0.0f;
    }, 4);

    const int runs = 200;
    const auto time = [&](const std::function<void()>& work) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < runs; i++) work();
        return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count() / runs;
    };

    Path path;
    std::vector<ObstacleMap::Spot> spots;
    int found = 0, foundFree = 0;
    float sum = 0;
    int ray = 0;
    const float planMs = time([&] { found += map.planPath(0, 0, 3, 0, 0.4f, path); });
    const float planFreeMs = time([&] { foundFree += map.planPath(0, 0, 2.5f, 1.6f, 0.4f, path); });
    const float spotsMs = time([&] { map.reachableSpots(0, 0, 1.5f, 8, spots); });
    const float freeMs = time([&] { sum += map.freeDistance(0, 0, 0.1f * (ray++ % 10), 3); });

    std::cout << "planPath (3m, person at goal): " << planMs << " ms, found " << found << "/" << runs << std::endl;
    std::cout << "planPath (3m, free goal): " << planFreeMs << " ms, found " << foundFree << "/" << runs << std::endl;
    std::cout << "reachableSpots (8m): " << spotsMs << " ms, " << spots.size() << " spots" << std::endl;
    std::cout << "freeDistance (3m): " << freeMs * 1000 << " us (sum " << sum << ")" << std::endl;
    return 0;
}
