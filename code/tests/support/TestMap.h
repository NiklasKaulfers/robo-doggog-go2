#ifndef FOLLOW_TESTS_TESTMAP_H
#define FOLLOW_TESTS_TESTMAP_H

// Synthetic LiDAR height maps for testing the ObstacleMap without the robot: the
// terrain is a function of the world position, a frame of it is cut out around the
// robot like the Go2 publishes it, and the map's clock is moved forward by hand.

#include <cmath>
#include <functional>
#include <utility>
#include <vector>
#include "core/ObstacleMap.h"

namespace testutil {

using robo::sdk::core::HeightMapFrame;
using robo::sdk::core::ObstacleMap;
using Path = std::vector<std::pair<float, float>>;
// terrain height at a world point, NAN where the LiDAR sees nothing
using Terrain = std::function<float(float, float)>;

// a map with a fake clock that the test moves forward by hand
struct TestMap {
    ObstacleMap map;
    float time = 0;

    TestMap() { map.setClock([this] { return time; }); }
    TestMap(const TestMap&) = delete;
    TestMap& operator=(const TestMap&) = delete;

    // one LiDAR frame of the given terrain, centered on (cx, cy), 2 * half meters wide
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
                data[static_cast<size_t>(y) * n + x] = std::isnan(h) ? 1e9f : h;
            }
        }
        frame.data = data.data();
        frame.size = data.size();
        map.integrate(frame);
    }
};

inline float flat(float, float) { return 0; }

inline bool inBox(float x, float y, float x0, float x1, float y0, float y1) {
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

// whether any route point lies in the region
inline bool touches(const Path& path, const std::function<bool(float, float)>& region) {
    for (const auto& [x, y] : path)
        if (region(x, y)) return true;
    return false;
}

// the route ends at a cell center within radius of the goal's cell, so allow half a cell
inline bool endsNear(const Path& path, float x, float y, float radius) {
    return !path.empty() && std::hypot(path.back().first - x, path.back().second - y) <= radius + 0.08f;
}

// which side (+1 left, -1 right) the route passes an obstacle at x
inline int sideAt(const Path& path, float x) {
    float best = 1e9f, y = 0;
    for (const auto& p : path) {
        if (std::abs(p.first - x) < best) {
            best = std::abs(p.first - x);
            y = p.second;
        }
    }
    return y >= 0 ? 1 : -1;
}
}

#endif //FOLLOW_TESTS_TESTMAP_H
