#include "ObstacleMapSubscriber.h"

#include <cmath>
#include <functional>
#include <iostream>

#define TOPIC_HEIGHT_MAP "rt/utlidar/height_map_array"

namespace robo::sdk::core
{
    ObstacleMapSubscriber::ObstacleMapSubscriber(ObstacleMap& map, const Control& control) : map(map), control(control)
    {
    }

    void ObstacleMapSubscriber::init()
    {
        subMap.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::HeightMap_>(TOPIC_HEIGHT_MAP));
        subMap->InitChannel(std::bind(&ObstacleMapSubscriber::onMessage, this, std::placeholders::_1), 1);
    }

    void ObstacleMapSubscriber::onMessage(const void* message)
    {
        const auto& msg = *static_cast<const unitree_go::msg::dds_::HeightMap_*>(message);
        if (!received) {
            received = true;
            std::cout << "[ObstacleMap] receiving height map, frame: " << msg.frame_id() << std::endl;
        }

        HeightMapFrame frame;
        frame.resolution = msg.resolution();
        frame.width = static_cast<int>(msg.width());
        frame.height = static_cast<int>(msg.height());
        frame.originX = msg.origin()[0];
        frame.originY = msg.origin()[1];
        frame.data = msg.data().data();
        frame.size = msg.data().size();
        map.integrate(frame);
        // warns once if the map is far from where odometry puts the robot; only once the
        // robot state arrived, before that the position reads 0, 0
        if (std::isfinite(control.stateAge()))
            map.checkFrame(control.positionX(), control.positionY());
    }
}
