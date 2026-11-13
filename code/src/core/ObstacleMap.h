#ifndef FOLLOW_OBSTACLEMAP_H
#define FOLLOW_OBSTACLEMAP_H

#include "unitree/idl/go2/HeightMap_.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <chrono>
#include <cmath>
#include <mutex>
#include <utility>
#include <vector>

namespace robo::sdk::core
{
// Remembered terrain around the robot. The Go2's LiDAR only publishes a small height
// map around the robot; it is copied into a larger grid that keeps what was seen for a
// few minutes, so routes can be planned through areas that are out of sensor range now.
class ObstacleMap
{
    struct Cell
    {
        float height = NAN;
        // time the cell was last observed, seconds since start
        float seen = -1e9f;
    };

    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::HeightMap_> subMap;
    // memory grid, row-major, follows the robot around
    std::vector<Cell> cells;
    float originX = 0, originY = 0;
    bool hasMap = false;
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    mutable std::mutex mapMutex;

public:

    void init();
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

    void updateMap(const void* message);
    void recenter(float x, float y);
    float now() const;
    bool cellIndex(float x, float y, int& ix, int& iy) const;
    // remembered height of a cell, or NAN if unknown or forgotten
    float heightAt(int ix, int iy, float maxAge) const;
    // cost of walking onto each cell: INFINITY for walls, UNKNOWN_COST for unseen terrain
    void buildCostGrid(std::vector<float>& cost) const;
    // marks the area under and right around the robot as walkable
    void clearAroundRobot(std::vector<float>& cost, int sx, int sy) const;
};
}

#endif //FOLLOW_OBSTACLEMAP_H
