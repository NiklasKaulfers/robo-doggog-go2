#include "ObstacleMap.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <queue>

// a height change larger than this between neighbouring cells can't be stepped over.
// regular stair steps are ~0.15-0.18m, walls/furniture/people are much higher
static constexpr float MAX_STEP_HEIGHT = 0.22f;
// half the robot's width plus some margin; rays are cast at the center and both sides
static constexpr float HALF_ROBOT_WIDTH = 0.2f;
// skip the area under the robot's own body
static constexpr float RAY_START = 0.3f;
// the ground the robot stands on is taken from the ring of cells just outside its body
// (median of the fresh ones). An obstacle that already starts inside RAY_START then
// still shows up as a jump at the first cell of a ray
static constexpr float GROUND_RING_INNER = 0.3f;
static constexpr float GROUND_RING_OUTER = 0.5f;
static constexpr int GROUND_MIN_CELLS = 10;
// cells searched around the robot for that ring, covers GROUND_RING_OUTER with margin
static constexpr int GROUND_SEARCH_CELLS = 7;

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
// around the robot the LiDAR can't see the ground (blind zone) and may see the robot's own
// legs; this area is treated as free so planning can always leave the robot's position
static constexpr int ROBOT_CLEAR_CELLS = 5;
// limits the planning time if the goal can't be reached
static constexpr int MAX_EXPANSIONS = 60000;
// routes are searched in a window around start and goal plus this margin (m), not in the
// whole 40m grid; detours longer than that aren't useful for following anyway
static constexpr float PLAN_MARGIN = 5.0f;
// the goal is often an obstacle itself (the person's legs). Near it, cells that are only
// blocked because obstacles were grown by the robot's size get this cost instead, so the
// route can end next to the person. The obstacle cells themselves stay blocked
static constexpr float GOAL_INFLATED_COST = 10.0f;
// hysteresis: cells along the last route are this much cheaper, so two about equally
// long detours don't flip every tick. Only while the last route is recent and went to
// about the same goal
static constexpr float LAST_PATH_DISCOUNT = 0.8f;
static constexpr int LAST_PATH_CELLS = 1;
static constexpr float LAST_PATH_TIME = 2.0f;
static constexpr float LAST_PATH_GOAL_DISTANCE = 1.0f;
// the LiDAR map is centered on the robot; further than this from the odometry position
// means the two use different frames (m)
static constexpr float FRAME_MISMATCH_DISTANCE = 2.0f;

namespace robo::sdk::core
{
    ObstacleMap::ObstacleMap() : cells(SIZE * SIZE, Cell {})
    {
    }

    void ObstacleMap::setClock(Clock newClock)
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        clock = std::move(newClock);
    }

    float ObstacleMap::now() const
    {
        if (clock) return clock();
        return std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
    }

    void ObstacleMap::integrate(const HeightMapFrame& frame)
    {
        const float res = frame.resolution;
        if (res <= 0 || frame.width <= 0 || frame.height <= 0 || frame.data == nullptr) return;

        std::lock_guard<std::mutex> lock(mapMutex);
        // the local map is centered on the robot
        const float centerX = frame.originX + frame.width * res / 2;
        const float centerY = frame.originY + frame.height * res / 2;
        const float memCenterX = originX + SIZE * RESOLUTION / 2;
        const float memCenterY = originY + SIZE * RESOLUTION / 2;
        if (!hasMap || std::hypot(centerX - memCenterX, centerY - memCenterY) > RECENTER_DISTANCE)
            recenter(centerX, centerY);
        hasMap = true;

        // copy every memory cell covered by the local map; cells the LiDAR doesn't see
        // right now keep their remembered height
        const float t = now();
        lastUpdate = t;
        frameCenterX = centerX;
        frameCenterY = centerY;
        int ix0, iy0, ix1, iy1;
        cellIndex(frame.originX, frame.originY, ix0, iy0);
        cellIndex(frame.originX + frame.width * res, frame.originY + frame.height * res, ix1, iy1);
        for (int iy = std::max(iy0, 0); iy <= std::min(iy1, SIZE - 1); iy++) {
            for (int ix = std::max(ix0, 0); ix <= std::min(ix1, SIZE - 1); ix++) {
                const float wx = originX + (ix + 0.5f) * RESOLUTION;
                const float wy = originY + (iy + 0.5f) * RESOLUTION;
                const int mx = static_cast<int>(std::floor((wx - frame.originX) / res));
                const int my = static_cast<int>(std::floor((wy - frame.originY) / res));
                if (mx < 0 || my < 0 || mx >= frame.width || my >= frame.height) continue;

                const size_t idx = static_cast<size_t>(my) * frame.width + mx;
                if (idx >= frame.size) continue;
                const float h = frame.data[idx];
                // unobserved cells are filled with huge values
                if (!std::isfinite(h) || std::abs(h) >= 100.0f) continue;

                cells[iy * SIZE + ix] = Cell {h, t};
            }
        }
    }

    float ObstacleMap::secondsSinceUpdate() const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap) return INFINITY;
        return now() - lastUpdate;
    }

    bool ObstacleMap::checkFrame(float robotX, float robotY)
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap) return true;
        const float offset = std::hypot(frameCenterX - robotX, frameCenterY - robotY);
        if (offset <= FRAME_MISMATCH_DISTANCE) return true;
        if (!frameWarned) {
            frameWarned = true;
            std::cerr << "[ObstacleMap] height map is centered at " << frameCenterX << ", " << frameCenterY
                      << " but the robot is at " << robotX << ", " << robotY << " (" << offset
                      << "m apart): height map and odometry seem to use different frames, routes will be wrong"
                      << std::endl;
        }
        return false;
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

    bool ObstacleMap::contains(float x, float y) const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        int ix, iy;
        return hasMap && cellIndex(x, y, ix, iy);
    }

    bool ObstacleMap::cellIndex(float x, float y, int& ix, int& iy) const
    {
        ix = static_cast<int>(std::floor((x - originX) / RESOLUTION));
        iy = static_cast<int>(std::floor((y - originY) / RESOLUTION));
        return ix >= 0 && iy >= 0 && ix < SIZE && iy < SIZE;
    }

    float ObstacleMap::heightAt(int ix, int iy, float maxAge, float t) const
    {
        if (ix < 0 || iy < 0 || ix >= SIZE || iy >= SIZE) return NAN;
        const Cell& cell = cells[iy * SIZE + ix];
        return (t - cell.seen <= maxAge) ? cell.height : NAN;
    }

    float ObstacleMap::groundLevel(float x, float y, float t) const
    {
        int cx, cy;
        cellIndex(x, y, cx, cy);
        constexpr int r = GROUND_SEARCH_CELLS;
        std::array<float, (2 * r + 1) * (2 * r + 1)> heights {};
        int count = 0;
        for (int iy = cy - r; iy <= cy + r; iy++) {
            for (int ix = cx - r; ix <= cx + r; ix++) {
                const float d = std::hypot(originX + (ix + 0.5f) * RESOLUTION - x, originY + (iy + 0.5f) * RESOLUTION - y);
                if (d < GROUND_RING_INNER || d > GROUND_RING_OUTER) continue;
                const float h = heightAt(ix, iy, FRESH_TIME, t);
                if (!std::isnan(h)) heights[count++] = h;
            }
        }
        if (count < GROUND_MIN_CELLS) return NAN;
        std::nth_element(heights.begin(), heights.begin() + count / 2, heights.begin() + count);
        return heights[count / 2];
    }

    float ObstacleMap::freeDistance(float x, float y, float heading, float maxDistance) const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap) return maxDistance;
        const float t = now();

        const float dx = std::cos(heading), dy = std::sin(heading);
        // perpendicular offset for the side rays
        const float px = -dy * HALF_ROBOT_WIDTH, py = dx * HALF_ROBOT_WIDTH;
        // the first known cell of each ray is compared with the ground around the robot;
        // without enough known ground it is only compared along the ray, as before
        const float ground = groundLevel(x, y, t);

        float freeDist = maxDistance;
        for (const float side : {-1.0f, 0.0f, 1.0f}) {
            const float ox = x + side * px, oy = y + side * py;
            float lastHeight = ground;
            for (float d = RAY_START; d < freeDist; d += RESOLUTION) {
                int ix, iy;
                cellIndex(ox + dx * d, oy + dy * d, ix, iy);
                // only recent terrain: whatever moved away since shouldn't stop the robot
                const float h = heightAt(ix, iy, FRESH_TIME, t);
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

    void ObstacleMap::buildCostWindow(int x0, int y0, int x1, int y1) const
    {
        x0 = std::max(x0, 0);
        y0 = std::max(y0, 0);
        x1 = std::min(x1, SIZE - 1);
        y1 = std::min(y1, SIZE - 1);
        // obstacles up to INFLATE_CELLS outside the window still grow into it
        const int ex0 = std::max(x0 - INFLATE_CELLS, 0), ey0 = std::max(y0 - INFLATE_CELLS, 0);
        const int ex1 = std::min(x1 + INFLATE_CELLS, SIZE - 1), ey1 = std::min(y1 + INFLATE_CELLS, SIZE - 1);
        const int ew = ex1 - ex0 + 1, eh = ey1 - ey0 + 1;
        window = Window {x0, y0, x1 - x0 + 1, y1 - y0 + 1, ex0, ey0, ew, originX, originY};
        const int w = window.w, h = window.h;
        const float t = now();
        rawObstacle.assign(static_cast<size_t>(ew) * eh, 0.0f);
        const auto known = [&](const Cell& cell) { return t - cell.seen <= MEMORY_TIME && !std::isnan(cell.height); };
        const auto mark = [&](int ix, int iy, float c) {
            if (ix < ex0 || iy < ey0 || ix > ex1 || iy > ey1) return;
            float& o = rawObstacle[(iy - ey0) * ew + (ix - ex0)];
            o = std::max(o, c);
        };
        // edges are checked from one cell before the area so its border isn't missed
        for (int iy = std::max(ey0 - 1, 0); iy <= ey1; iy++) {
            for (int ix = std::max(ex0 - 1, 0); ix <= ex1; ix++) {
                const Cell& cell = cells[iy * SIZE + ix];
                if (!known(cell)) continue;
                // an edge too high to step over makes both cells an obstacle
                for (const auto& [nx, ny] : {std::pair {ix + 1, iy}, std::pair {ix, iy + 1}}) {
                    if (nx >= SIZE || ny >= SIZE) continue;
                    const Cell& other = cells[ny * SIZE + nx];
                    if (!known(other) || std::abs(other.height - cell.height) <= MAX_STEP_HEIGHT) continue;

                    const float seen = std::max(cell.seen, other.seen);
                    const float c = (t - seen <= FRESH_TIME) ? INFINITY : OLD_OBSTACLE_COST;
                    mark(ix, iy, c);
                    mark(nx, ny, c);
                }
            }
        }

        cost.resize(static_cast<size_t>(w) * h);
        for (int ly = 0; ly < h; ly++)
            for (int lx = 0; lx < w; lx++)
                cost[ly * w + lx] = known(cells[(y0 + ly) * SIZE + x0 + lx]) ? 1.0f : UNKNOWN_COST;

        // grow obstacles by the robot's size
        for (int iy = ey0; iy <= ey1; iy++) {
            for (int ix = ex0; ix <= ex1; ix++) {
                const float c = rawObstacle[(iy - ey0) * ew + (ix - ex0)];
                if (c == 0) continue;
                for (int oy = std::max(iy - INFLATE_CELLS, y0); oy <= std::min(iy + INFLATE_CELLS, y1); oy++) {
                    for (int ox = std::max(ix - INFLATE_CELLS, x0); ox <= std::min(ix + INFLATE_CELLS, x1); ox++) {
                        if ((ox - ix) * (ox - ix) + (oy - iy) * (oy - iy) > INFLATE_CELLS * INFLATE_CELLS) continue;
                        float& target = cost[(oy - y0) * w + (ox - x0)];
                        target = std::max(target, c);
                    }
                }
            }
        }
    }

    void ObstacleMap::clearAroundRobot(int sx, int sy) const
    {
        for (int iy = std::max(sy - ROBOT_CLEAR_CELLS, 0); iy <= std::min(sy + ROBOT_CLEAR_CELLS, window.h - 1); iy++) {
            for (int ix = std::max(sx - ROBOT_CLEAR_CELLS, 0); ix <= std::min(sx + ROBOT_CLEAR_CELLS, window.w - 1); ix++) {
                if ((ix - sx) * (ix - sx) + (iy - sy) * (iy - sy) > ROBOT_CLEAR_CELLS * ROBOT_CLEAR_CELLS) continue;
                cost[iy * window.w + ix] = 1.0f;
            }
        }
    }

    void ObstacleMap::relaxAroundGoal(int gx, int gy, float goalCells) const
    {
        // the goal area plus the distance obstacles were grown by, so the route can get
        // from free terrain into the goal area
        const float radius = goalCells + INFLATE_CELLS;
        const int r = static_cast<int>(std::ceil(radius));
        for (int iy = std::max(gy - r, 0); iy <= std::min(gy + r, window.h - 1); iy++) {
            for (int ix = std::max(gx - r, 0); ix <= std::min(gx + r, window.w - 1); ix++) {
                if (std::hypot(float(ix - gx), float(iy - gy)) > radius) continue;
                const int idx = iy * window.w + ix;
                const float raw = rawObstacle[(window.y0 + iy - window.ey0) * window.ew + (window.x0 + ix - window.ex0)];
                if (std::isinf(cost[idx]) && raw == 0) cost[idx] = GOAL_INFLATED_COST;
            }
        }
    }

    void ObstacleMap::preferLastPath(float toX, float toY, float t) const
    {
        if (lastPath.empty() || t - lastPathTime > LAST_PATH_TIME) return;
        if (std::hypot(lastPath.back().first - toX, lastPath.back().second - toY) > LAST_PATH_GOAL_DISTANCE) return;

        // visited marks cells already discounted, so overlapping points don't add up
        visited.assign(cost.size(), 0);
        for (const auto& [px, py] : lastPath) {
            const int cx = static_cast<int>(std::floor((px - window.originX) / RESOLUTION)) - window.x0;
            const int cy = static_cast<int>(std::floor((py - window.originY) / RESOLUTION)) - window.y0;
            for (int iy = std::max(cy - LAST_PATH_CELLS, 0); iy <= std::min(cy + LAST_PATH_CELLS, window.h - 1); iy++) {
                for (int ix = std::max(cx - LAST_PATH_CELLS, 0); ix <= std::min(cx + LAST_PATH_CELLS, window.w - 1); ix++) {
                    const int idx = iy * window.w + ix;
                    // blocked or expensive cells stay as they are
                    if (visited[idx] || cost[idx] > UNKNOWN_COST) continue;
                    visited[idx] = 1;
                    cost[idx] *= LAST_PATH_DISCOUNT;
                }
            }
        }
    }

    bool ObstacleMap::planPath(float fromX, float fromY, float toX, float toY, float goalRadius,
                               std::vector<std::pair<float, float>>& path) const
    {
        path.clear();
        std::lock_guard<std::mutex> planLock(planMutex);
        float t;
        int sx, sy, gx, gy;
        {
            // the map lock is only held while the cost window is copied out of the grid
            std::lock_guard<std::mutex> lock(mapMutex);
            if (!hasMap) return false;
            if (!cellIndex(fromX, fromY, sx, sy) || !cellIndex(toX, toY, gx, gy)) return false;
            t = now();
            const int margin = static_cast<int>(std::ceil(PLAN_MARGIN / RESOLUTION));
            buildCostWindow(std::min(sx, gx) - margin, std::min(sy, gy) - margin,
                            std::max(sx, gx) + margin, std::max(sy, gy) + margin);
        }
        const int w = window.w, h = window.h;
        sx -= window.x0;
        sy -= window.y0;
        gx -= window.x0;
        gy -= window.y0;

        const float goalCells = goalRadius / RESOLUTION;
        relaxAroundGoal(gx, gy, goalCells);
        preferLastPath(toX, toY, t);
        clearAroundRobot(sx, sy);

        // A* over the 8-connected grid
        const auto heuristic = [&](int ix, int iy) {
            return std::max(0.0f, std::hypot(float(ix - gx), float(iy - gy)) - goalCells);
        };
        dist.assign(static_cast<size_t>(w) * h, INFINITY);
        parent.assign(static_cast<size_t>(w) * h, -1);
        using Node = std::pair<float, int>;
        std::priority_queue<Node, std::vector<Node>, std::greater<>> open;

        const int start = sy * w + sx;
        dist[start] = 0;
        open.push({heuristic(sx, sy), start});

        int goal = -1;
        int expansions = 0;
        while (!open.empty() && expansions++ < MAX_EXPANSIONS) {
            const auto [f, idx] = open.top();
            open.pop();
            const int ix = idx % w, iy = idx / w;
            if (f > dist[idx] + heuristic(ix, iy) + 1e-3f) continue;

            if (std::hypot(float(ix - gx), float(iy - gy)) <= goalCells) {
                goal = idx;
                break;
            }

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    const int nx = ix + dx, ny = iy + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const int n = ny * w + nx;
                    if (std::isinf(cost[n])) continue;
                    const bool diagonal = dx != 0 && dy != 0;
                    // no diagonal steps past the corner of a blocked cell
                    if (diagonal && (std::isinf(cost[iy * w + nx]) || std::isinf(cost[ny * w + ix]))) continue;

                    const float d = dist[idx] + (diagonal ? 1.414f : 1.0f) * cost[n];
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
            path.emplace_back(window.originX + (window.x0 + idx % w + 0.5f) * RESOLUTION,
                              window.originY + (window.y0 + idx / w + 0.5f) * RESOLUTION);
        std::reverse(path.begin(), path.end());
        lastPath = path;
        lastPathTime = t;
        return true;
    }

    void ObstacleMap::reachableSpots(float fromX, float fromY, float minDistance, float maxDistance,
                                     std::vector<Spot>& spots) const
    {
        spots.clear();
        std::lock_guard<std::mutex> planLock(planMutex);
        int sx, sy;
        {
            std::lock_guard<std::mutex> lock(mapMutex);
            if (!hasMap) return;
            if (!cellIndex(fromX, fromY, sx, sy)) return;
            // nothing further than maxDistance is visited, one more cell for the frontier test
            const int r = static_cast<int>(std::ceil(maxDistance / RESOLUTION)) + 1;
            buildCostWindow(sx - r, sy - r, sx + r, sy + r);
        }
        const int w = window.w, h = window.h;
        sx -= window.x0;
        sy -= window.y0;
        clearAroundRobot(sx, sy);
        const auto unknown = [&](int ix, int iy) {
            return ix >= 0 && iy >= 0 && ix < w && iy < h && cost[iy * w + ix] == UNKNOWN_COST;
        };
        const auto worldX = [&](int ix) { return window.originX + (window.x0 + ix + 0.5f) * RESOLUTION; };
        const auto worldY = [&](int iy) { return window.originY + (window.y0 + iy + 0.5f) * RESOLUTION; };

        // breadth-first search through known, walkable terrain
        visited.assign(static_cast<size_t>(w) * h, 0);
        std::queue<int> open;
        open.push(sy * w + sx);
        visited[sy * w + sx] = 1;

        while (!open.empty()) {
            const int idx = open.front();
            open.pop();
            const int ix = idx % w, iy = idx / w;
            const float wx = worldX(ix);
            const float wy = worldY(iy);
            const float distance = std::hypot(wx - fromX, wy - fromY);

            // keep every 5th grid cell in each direction, enough choice without huge lists
            // goals only on known terrain
            if (distance >= minDistance && cost[idx] == 1.0f && (window.x0 + ix) % 5 == 0 && (window.y0 + iy) % 5 == 0) {
                const bool frontier = unknown(ix + 1, iy) || unknown(ix - 1, iy) || unknown(ix, iy + 1) || unknown(ix, iy - 1);
                spots.push_back(Spot {wx, wy, frontier});
            }

            for (const auto& [nx, ny] : {std::pair {ix + 1, iy}, std::pair {ix - 1, iy},
                                         std::pair {ix, iy + 1}, std::pair {ix, iy - 1}}) {
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                const int n = ny * w + nx;
                // through known or unseen terrain, but not through obstacles (old obstacles
                // count as blocked too, goals shouldn't be behind them)
                if (visited[n] || cost[n] > UNKNOWN_COST) continue;
                if (std::hypot(worldX(nx) - fromX, worldY(ny) - fromY) > maxDistance) continue;
                visited[n] = 1;
                open.push(n);
            }
        }
    }
}
