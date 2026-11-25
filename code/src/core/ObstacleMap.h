#ifndef FOLLOW_OBSTACLEMAP_H
#define FOLLOW_OBSTACLEMAP_H

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace robo::sdk::core
{
// one local height map as the LiDAR publishes it (rt/utlidar/height_map_array), without
// any SDK types so the map logic can be built and tested anywhere.
// data is row-major, width cells per row; cells nobody observed are huge (|h| >= 100) or
// not finite
struct HeightMapFrame
{
    float resolution = 0;
    int width = 0, height = 0;
    // world position of the corner of cell (0, 0)
    float originX = 0, originY = 0;
    const float* data = nullptr;
    size_t size = 0;
};

// Remembered terrain around the robot. The Go2's LiDAR only publishes a small height
// map around the robot; it is copied into a larger grid that keeps what was seen for a
// few minutes, so routes can be planned through areas that are out of sensor range now.
// Thread safe: frames may be integrated from the DDS thread while the trick loop queries.
// The DDS subscription itself lives in ObstacleMapSubscriber (app only).
class ObstacleMap
{
public:
    // seconds since some fixed start, replaceable so tests are deterministic
    using Clock = std::function<float()>;

private:
    struct Cell
    {
        float height = NAN;
        // time the cell was last observed, seconds since start
        float seen = -1e9f;
    };

    // memory grid, row-major, follows the robot around
    std::vector<Cell> cells;
    float originX = 0, originY = 0;
    bool hasMap = false;
    // when the last frame was integrated, and the center of that frame
    float lastUpdate = -1e9f;
    float frameCenterX = 0, frameCenterY = 0;
    bool frameWarned = false;
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    Clock clock;
    mutable std::mutex mapMutex;

    // planning scratch space, reused between calls. Guarded by planMutex, so the DDS
    // thread only waits while the cost window is copied out of the grid, not while A* runs
    mutable std::mutex planMutex;
    struct Window
    {
        // grid cell of the window's corner and its size in cells
        int x0 = 0, y0 = 0, w = 0, h = 0;
        // the larger area rawObstacle covers (window plus INFLATE_CELLS)
        int ex0 = 0, ey0 = 0, ew = 0;
        // origin of the memory grid when the window was built
        float originX = 0, originY = 0;
    };
    mutable Window window;
    mutable std::vector<float> cost;
    // obstacle cost before growing it by the robot's size, 0 = none (extended window)
    mutable std::vector<float> rawObstacle;
    mutable std::vector<float> dist;
    mutable std::vector<int> parent;
    mutable std::vector<unsigned char> visited;
    // the last planned route (world points) and when it was planned, for hysteresis
    mutable std::vector<std::pair<float, float>> lastPath;
    mutable float lastPathTime = -1e9f;

public:

    ObstacleMap();

    // replaces the time source (seconds), e.g. with a fake clock in tests
    void setClock(Clock newClock);

    // copies a local height map into the memory grid
    void integrate(const HeightMapFrame& frame);
    // seconds since the last frame was integrated, INFINITY before the first one
    float secondsSinceUpdate() const;
    // compares the center of the last frame with the robot's odometry position. They
    // should be close (the LiDAR map is centered on the robot); if not, the height map
    // and odometry use different frames and every plan would be wrong. Warns once and
    // returns false then; true if consistent or no frame yet
    bool checkFrame(float robotX, float robotY);

    bool available() const;
    // whether a world position lies inside the remembered area
    bool contains(float x, float y) const;

    // free distance (meters) from (x, y) along a world heading, up to maxDistance, only
    // looking at recently seen terrain. A path is blocked by height jumps a leg can't
    // step over; stair steps stay free.
    float freeDistance(float x, float y, float heading, float maxDistance) const;

    // plans a route through the remembered terrain from (fromX, fromY) to within
    // goalRadius of (toX, toY). path holds world points, start first.
    bool planPath(float fromX, float fromY, float toX, float toY, float goalRadius,
                  std::vector<std::pair<float, float>>& path) const;

    struct Spot
    {
        float x, y;
        // at the edge of the known map, next to unseen terrain
        bool frontier;
    };

    // known, walkable spots that can be reached from (fromX, fromY) without crossing an
    // obstacle, between minDistance and maxDistance away, about every 0.5m
    void reachableSpots(float fromX, float fromY, float minDistance, float maxDistance,
                        std::vector<Spot>& spots) const;

private:

    void recenter(float x, float y);
    float now() const;
    bool cellIndex(float x, float y, int& ix, int& iy) const;
    // remembered height of a cell, or NAN if unknown or forgotten
    float heightAt(int ix, int iy, float maxAge, float t) const;
    // ground level around (x, y) just outside the robot's body, NAN if not enough is known
    float groundLevel(float x, float y, float t) const;
    // fills cost/rawObstacle for the grid cells [x0, x1] x [y0, y1] (clipped to the grid):
    // INFINITY for walls, UNKNOWN_COST for unseen terrain. Needs mapMutex
    void buildCostWindow(int x0, int y0, int x1, int y1) const;
    // marks the area under and right around the robot as walkable (window cell coords)
    void clearAroundRobot(int sx, int sy) const;
    // lets the route reach a goal that sits next to an obstacle (window cell coords)
    void relaxAroundGoal(int gx, int gy, float goalCells) const;
    // makes cells along the previous route a little cheaper so the route doesn't flip
    void preferLastPath(float toX, float toY, float t) const;
};
}

#endif //FOLLOW_OBSTACLEMAP_H
