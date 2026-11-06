#include "ObstacleMap.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>

#define TOPIC_HEIGHT_MAP "rt/utlidar/height_map_array"

// a height change larger than this between neighbouring cells can't be stepped over.
// regular stair steps are ~0.15-0.18m, walls/furniture/people are much higher
static constexpr float MAX_STEP_HEIGHT = 0.22f;
// half the robot's width plus some margin; rays are cast at the center and both sides
static constexpr float HALF_ROBOT_WIDTH = 0.2f;
// skip the area under the robot's own body
static constexpr float RAY_START = 0.3f;

// memory grid: 400 x 400 cells of 10cm = 40m x 40m around the robot (~1.3MB)
static constexpr float RESOLUTION = 0.1f;
static constexpr int SIZE = 400;
// the grid is moved along once the robot gets this far from its center
static constexpr float RECENTER_DISTANCE = 10.0f;
// terrain is forgotten after this long without seeing it again (s)
static constexpr float MEMORY_TIME = 180.0f;
// obstacles seen within this time are walls; older ones may have been people that
// walked away, so they only make a route more expensive instead of blocking it (s)
static constexpr float FRESH_TIME = 10.0f;
static constexpr float OLD_OBSTACLE_COST = 20.0f;
// unseen terrain is allowed but routes through known free terrain are preferred
static constexpr float UNKNOWN_COST = 2.0f;
// obstacles are grown by this many cells so the robot's body keeps clear of them
static constexpr int INFLATE_CELLS = 3;
// limits the planning time if the goal can't be reached
static constexpr int MAX_EXPANSIONS = 60000;

namespace robo::sdk::core
{
    void ObstacleMap::init()
    {
        cells.assign(SIZE * SIZE, Cell {});
        subMap.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::HeightMap_>(TOPIC_HEIGHT_MAP));
        subMap->InitChannel(std::bind(&ObstacleMap::updateMap, this, std::placeholders::_1), 1);
    }

    float ObstacleMap::now() const
    {
        return std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
    }

    void ObstacleMap::updateMap(const void* message)
    {
        const auto& map = *static_cast<const unitree_go::msg::dds_::HeightMap_*>(message);
        const float res = map.resolution();
        if (res <= 0 || map.width() == 0 || map.height() == 0) return;

        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap)
            std::cout << "[ObstacleMap] receiving height map, frame: " << map.frame_id() << std::endl;

        // the local map is centered on the robot
        const float centerX = map.origin()[0] + map.width() * res / 2;
        const float centerY = map.origin()[1] + map.height() * res / 2;
        const float memCenterX = originX + SIZE * RESOLUTION / 2;
        const float memCenterY = originY + SIZE * RESOLUTION / 2;
        if (!hasMap || std::hypot(centerX - memCenterX, centerY - memCenterY) > RECENTER_DISTANCE)
            recenter(centerX, centerY);
        hasMap = true;

        // copy every memory cell covered by the local map; cells the LiDAR doesn't see
        // right now keep their remembered height
        const float t = now();
        int ix0, iy0, ix1, iy1;
        cellIndex(map.origin()[0], map.origin()[1], ix0, iy0);
        cellIndex(map.origin()[0] + map.width() * res, map.origin()[1] + map.height() * res, ix1, iy1);
        for (int iy = std::max(iy0, 0); iy <= std::min(iy1, SIZE - 1); iy++) {
            for (int ix = std::max(ix0, 0); ix <= std::min(ix1, SIZE - 1); ix++) {
                const float wx = originX + (ix + 0.5f) * RESOLUTION;
                const float wy = originY + (iy + 0.5f) * RESOLUTION;
                const int mx = static_cast<int>(std::floor((wx - map.origin()[0]) / res));
                const int my = static_cast<int>(std::floor((wy - map.origin()[1]) / res));
                if (mx < 0 || my < 0 || mx >= static_cast<int>(map.width()) || my >= static_cast<int>(map.height()))
                    continue;

                const size_t idx = static_cast<size_t>(my) * map.width() + mx;
                if (idx >= map.data().size()) continue;
                const float h = map.data()[idx];
                // unobserved cells are filled with huge values
                if (!std::isfinite(h) || std::abs(h) >= 100.0f) continue;

                cells[iy * SIZE + ix] = Cell {h, t};
            }
        }
    }

    void ObstacleMap::recenter(float x, float y)
    {
        // keep the grid aligned to whole cells so remembered cells don't shift
        const float newOriginX = std::floor(x / RESOLUTION) * RESOLUTION - SIZE * RESOLUTION / 2;
        const float newOriginY = std::floor(y / RESOLUTION) * RESOLUTION - SIZE * RESOLUTION / 2;
        const int shiftX = static_cast<int>(std::lround((newOriginX - originX) / RESOLUTION));
        const int shiftY = static_cast<int>(std::lround((newOriginY - originY) / RESOLUTION));

        std::vector<Cell> moved(SIZE * SIZE, Cell {});
        if (hasMap) {
            for (int iy = 0; iy < SIZE; iy++) {
                const int oy = iy + shiftY;
                if (oy < 0 || oy >= SIZE) continue;
                for (int ix = 0; ix < SIZE; ix++) {
                    const int ox = ix + shiftX;
                    if (ox < 0 || ox >= SIZE) continue;
                    moved[iy * SIZE + ix] = cells[oy * SIZE + ox];
                }
            }
        }
        cells.swap(moved);
        originX = newOriginX;
        originY = newOriginY;
    }

    bool ObstacleMap::available() const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        return hasMap;
    }

    bool ObstacleMap::cellIndex(float x, float y, int& ix, int& iy) const
    {
        ix = static_cast<int>(std::floor((x - originX) / RESOLUTION));
        iy = static_cast<int>(std::floor((y - originY) / RESOLUTION));
        return ix >= 0 && iy >= 0 && ix < SIZE && iy < SIZE;
    }

    float ObstacleMap::heightAt(int ix, int iy, float maxAge) const
    {
        if (ix < 0 || iy < 0 || ix >= SIZE || iy >= SIZE) return NAN;
        const Cell& cell = cells[iy * SIZE + ix];
        return (now() - cell.seen <= maxAge) ? cell.height : NAN;
    }

    float ObstacleMap::freeDistance(float x, float y, float heading, float maxDistance) const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap) return maxDistance;

        const float dx = std::cos(heading), dy = std::sin(heading);
        // perpendicular offset for the side rays
        const float px = -dy * HALF_ROBOT_WIDTH, py = dx * HALF_ROBOT_WIDTH;

        float freeDist = maxDistance;
        for (const float side : {-1.0f, 0.0f, 1.0f}) {
            const float ox = x + side * px, oy = y + side * py;
            float lastHeight = NAN;
            for (float d = RAY_START; d < freeDist; d += RESOLUTION) {
                int ix, iy;
                cellIndex(ox + dx * d, oy + dy * d, ix, iy);
                // only recent terrain: whatever moved away since shouldn't stop the robot
                const float h = heightAt(ix, iy, FRESH_TIME);
                // unknown cells are skipped, the avoidance service still guards them
                if (std::isnan(h)) continue;
                if (!std::isnan(lastHeight) && std::abs(h - lastHeight) > MAX_STEP_HEIGHT) {
                    freeDist = d;
                    break;
                }
                lastHeight = h;
            }
        }
        return freeDist;
    }

    void ObstacleMap::buildCostGrid(std::vector<float>& cost) const
    {
        const float t = now();
        cost.assign(SIZE * SIZE, 1.0f);
        std::vector<float> obstacle(SIZE * SIZE, 0.0f);
        for (int iy = 0; iy < SIZE; iy++) {
            for (int ix = 0; ix < SIZE; ix++) {
                const Cell& cell = cells[iy * SIZE + ix];
                const float h = (t - cell.seen <= MEMORY_TIME) ? cell.height : NAN;
                if (std::isnan(h)) {
                    cost[iy * SIZE + ix] = UNKNOWN_COST;
                    continue;
                }
                // an edge too high to step over makes this cell an obstacle
                for (const auto& [nx, ny] : {std::pair {ix + 1, iy}, std::pair {ix, iy + 1}}) {
                    if (nx >= SIZE || ny >= SIZE) continue;
                    const Cell& other = cells[ny * SIZE + nx];
                    if (t - other.seen > MEMORY_TIME || std::isnan(other.height)) continue;
                    if (std::abs(other.height - h) <= MAX_STEP_HEIGHT) continue;
    
                    const float seen = std::max(cell.seen, other.seen);
                    const float c = (t - seen <= FRESH_TIME) ? INFINITY : OLD_OBSTACLE_COST;
                    obstacle[iy * SIZE + ix] = std::max(obstacle[iy * SIZE + ix], c);
                    obstacle[ny * SIZE + nx] = std::max(obstacle[ny * SIZE + nx], c);
                }
            }
        }
        // grow obstacles by the robot's size
        for (int iy = 0; iy < SIZE; iy++) {
            for (int ix = 0; ix < SIZE; ix++) {
                const float c = obstacle[iy * SIZE + ix];
                if (c == 0) continue;
                for (int oy = std::max(iy - INFLATE_CELLS, 0); oy <= std::min(iy + INFLATE_CELLS, SIZE - 1); oy++) {
                    for (int ox = std::max(ix - INFLATE_CELLS, 0); ox <= std::min(ix + INFLATE_CELLS, SIZE - 1); ox++) {
                        if ((ox - ix) * (ox - ix) + (oy - iy) * (oy - iy) > INFLATE_CELLS * INFLATE_CELLS) continue;
                        cost[oy * SIZE + ox] = std::max(cost[oy * SIZE + ox], c);
                    }
                }
            }
        }
    }

    bool ObstacleMap::planPath(float fromX, float fromY, float toX, float toY, float goalRadius,
                               std::vector<std::pair<float, float>>& path) const
    {
        path.clear();
        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap) return false;

        int sx, sy, gx, gy;
        if (!cellIndex(fromX, fromY, sx, sy) || !cellIndex(toX, toY, gx, gy)) return false;

        std::vector<float> cost;
        buildCostGrid(cost);

        // the robot may already stand inside an inflated area, it has to be able to leave it
        cost[sy * SIZE + sx] = 1.0f;

        // A* over the 8-connected grid
        const float goalCells = goalRadius / RESOLUTION;
        const auto heuristic = [&](int ix, int iy) {
            return std::max(0.0f, std::hypot(float(ix - gx), float(iy - gy)) - goalCells);
        };
        std::vector<float> dist(SIZE * SIZE, INFINITY);
        std::vector<int> parent(SIZE * SIZE, -1);
        using Node = std::pair<float, int>;
        std::priority_queue<Node, std::vector<Node>, std::greater<>> open;

        const int start = sy * SIZE + sx;
        dist[start] = 0;
        open.push({heuristic(sx, sy), start});

        int goal = -1;
        int expansions = 0;
        while (!open.empty() && expansions++ < MAX_EXPANSIONS) {
            const auto [f, idx] = open.top();
            open.pop();
            const int ix = idx % SIZE, iy = idx / SIZE;
            if (f > dist[idx] + heuristic(ix, iy) + 1e-3f) continue;

            if (std::hypot(float(ix - gx), float(iy - gy)) <= goalCells) {
                goal = idx;
                break;
            }

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    const int nx = ix + dx, ny = iy + dy;
                    if (nx < 0 || ny < 0 || nx >= SIZE || ny >= SIZE) continue;
                    const int n = ny * SIZE + nx;
                    if (std::isinf(cost[n])) continue;

                    const float d = dist[idx] + ((dx != 0 && dy != 0) ? 1.414f : 1.0f) * cost[n];
                    if (d < dist[n]) {
                        dist[n] = d;
                        parent[n] = idx;
                        open.push({d + heuristic(nx, ny), n});
                    }
                }
            }
        }
        if (goal < 0) return false;

        for (int idx = goal; idx >= 0; idx = parent[idx])
            path.emplace_back(originX + (idx % SIZE + 0.5f) * RESOLUTION, originY + (idx / SIZE + 0.5f) * RESOLUTION);
        std::reverse(path.begin(), path.end());
        return true;
    }

    bool ObstacleMap::findFrontier(float fromX, float fromY, float minDistance,
                                   const std::vector<std::pair<float, float>>& ignore, float ignoreRadius,
                                   float& outX, float& outY) const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap) return false;

        int sx, sy;
        if (!cellIndex(fromX, fromY, sx, sy)) return false;

        std::vector<float> cost;
        buildCostGrid(cost);
        cost[sy * SIZE + sx] = 1.0f;
        const auto unknown = [&](int ix, int iy) {
            return ix >= 0 && iy >= 0 && ix < SIZE && iy < SIZE && cost[iy * SIZE + ix] == UNKNOWN_COST;
        };

        // breadth-first search through known, walkable terrain finds the nearest frontier
        std::vector<bool> visited(SIZE * SIZE, false);
        std::queue<int> open;
        open.push(sy * SIZE + sx);
        visited[sy * SIZE + sx] = true;

        while (!open.empty()) {
            const int idx = open.front();
            open.pop();
            const int ix = idx % SIZE, iy = idx / SIZE;
            const float wx = originX + (ix + 0.5f) * RESOLUTION;
            const float wy = originY + (iy + 0.5f) * RESOLUTION;

            const bool frontier = unknown(ix + 1, iy) || unknown(ix - 1, iy) || unknown(ix, iy + 1) || unknown(ix, iy - 1);
            if (frontier && std::hypot(wx - fromX, wy - fromY) >= minDistance) {
                const bool ignored = std::any_of(ignore.begin(), ignore.end(), [&](const auto& p) {
                    return std::hypot(p.first - wx, p.second - wy) < ignoreRadius;
                });
                if (!ignored) {
                    outX = wx;
                    outY = wy;
                    return true;
                }
            }

            for (const auto& [nx, ny] : {std::pair {ix + 1, iy}, std::pair {ix - 1, iy},
                                         std::pair {ix, iy + 1}, std::pair {ix, iy - 1}}) {
                if (nx < 0 || ny < 0 || nx >= SIZE || ny >= SIZE) continue;
                const int n = ny * SIZE + nx;
                // only walk through known terrain that isn't blocked by a recent obstacle
                if (visited[n] || std::isinf(cost[n]) || cost[n] == UNKNOWN_COST) continue;
                visited[n] = true;
                open.push(n);
            }
        }
        return false;
    }
}
