#ifndef FOLLOW_OBSTACLEMAPSUBSCRIBER_H
#define FOLLOW_OBSTACLEMAPSUBSCRIBER_H

#include "Control.h"
#include "ObstacleMap.h"
#include "unitree/idl/go2/HeightMap_.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace robo::sdk::core
{
// Feeds the LiDAR height map (DDS topic rt/utlidar/height_map_array) into an ObstacleMap.
// Kept apart from ObstacleMap so the map logic builds and is tested without the SDK.
class ObstacleMapSubscriber
{
    ObstacleMap& map;
    // odometry position, to warn when the height map uses another frame
    const Control& control;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::HeightMap_> subMap;
    bool received = false;

public:

    ObstacleMapSubscriber(ObstacleMap& map, const Control& control);

    // starts the subscription; DDS must be initialized
    void init();

private:

    void onMessage(const void* message);
};
}

#endif //FOLLOW_OBSTACLEMAPSUBSCRIBER_H
