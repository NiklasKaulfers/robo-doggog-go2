#ifndef FOLLOW_OBSTACLEMAP_H
#define FOLLOW_OBSTACLEMAP_H

#include "unitree/idl/go2/HeightMap_.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <mutex>

namespace robo::sdk::core
{
// Height map around the robot built by the Go2's LiDAR. Used to look ahead along a
// heading and find out how far the robot can walk before something blocks it.
class ObstacleMap
{
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::HeightMap_> subMap;
    unitree_go::msg::dds_::HeightMap_ map;
    bool hasMap = false;
    mutable std::mutex mapMutex;

public:

    void init();
    bool available() const;

    // free distance (meters) from (x, y) along a world heading, up to maxDistance.
    // A path is blocked by height jumps a leg can't step over; stair steps stay free.
    float freeDistance(float x, float y, float heading, float maxDistance) const;

private:

    void updateMap(const void* message);
    // terrain height at a world position, or NAN if the cell is unknown
    float heightAt(float x, float y) const;
};
}

#endif //FOLLOW_OBSTACLEMAP_H
