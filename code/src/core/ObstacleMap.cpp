#include "ObstacleMap.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#define TOPIC_HEIGHT_MAP "rt/utlidar/height_map_array"

// a height change larger than this between neighbouring cells can't be stepped over.
// regular stair steps are ~0.15-0.18m, walls/furniture/people are much higher
static constexpr float MAX_STEP_HEIGHT = 0.22f;
// half the robot's width plus some margin; rays are cast at the center and both sides
static constexpr float HALF_ROBOT_WIDTH = 0.2f;
// skip the area under the robot's own body
static constexpr float RAY_START = 0.3f;

namespace robo::sdk::core
{
    void ObstacleMap::init()
    {
        subMap.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::HeightMap_>(TOPIC_HEIGHT_MAP));
        subMap->InitChannel(std::bind(&ObstacleMap::updateMap, this, std::placeholders::_1), 1);
    }

    void ObstacleMap::updateMap(const void* message)
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        map = *static_cast<const unitree_go::msg::dds_::HeightMap_*>(message);
        if (!hasMap)
            std::cout << "[ObstacleMap] receiving height map, frame: " << map.frame_id() << std::endl;
        hasMap = true;
    }

    bool ObstacleMap::available() const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        return hasMap;
    }

    float ObstacleMap::heightAt(float x, float y) const
    {
        if (map.resolution() <= 0) return NAN;
        const int ix = static_cast<int>(std::floor((x - map.origin()[0]) / map.resolution()));
        const int iy = static_cast<int>(std::floor((y - map.origin()[1]) / map.resolution()));
        if (ix < 0 || iy < 0 || ix >= static_cast<int>(map.width()) || iy >= static_cast<int>(map.height()))
            return NAN;

        const size_t idx = static_cast<size_t>(iy) * map.width() + ix;
        if (idx >= map.data().size()) return NAN;
        const float h = map.data()[idx];
        // unobserved cells are filled with huge values
        return (std::isfinite(h) && std::abs(h) < 100.0f) ? h : NAN;
    }

    float ObstacleMap::freeDistance(float x, float y, float heading, float maxDistance) const
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        if (!hasMap) return maxDistance;

        const float dx = std::cos(heading), dy = std::sin(heading);
        // perpendicular offset for the side rays
        const float px = -dy * HALF_ROBOT_WIDTH, py = dx * HALF_ROBOT_WIDTH;
        const float step = std::max(map.resolution(), 0.02f);

        float freeDist = maxDistance;
        for (const float side : {-1.0f, 0.0f, 1.0f}) {
            const float ox = x + side * px, oy = y + side * py;
            float lastHeight = NAN;
            for (float d = RAY_START; d < freeDist; d += step) {
                const float h = heightAt(ox + dx * d, oy + dy * d);
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
}
