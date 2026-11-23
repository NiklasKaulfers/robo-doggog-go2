#ifndef FAKESDK_OBSTACLES_AVOID_CLIENT_HPP
#define FAKESDK_OBSTACLES_AVOID_CLIENT_HPP

// fake of the Unitree SDK's obstacle avoidance client, calls are recorded as "avoid.<Method>"
// and change the avoidance state on fakesdk::Bus like the real service would

#include <cstdint>
#include <unitree/robot/client/client.hpp>

namespace unitree::robot::go2 {
class ObstaclesAvoidClient : public Client {
public:
    ObstaclesAvoidClient();
    ~ObstaclesAvoidClient() override;

    void Init() override;

    int32_t SwitchSet(bool enable);
    int32_t SwitchGet(bool& enable);

    int32_t Move(float x, float y, float yaw);
    int32_t UseRemoteCommandFromApi(bool isRemoteCommandsFromApi);

    int32_t MoveToAbsolutePosition(float x, float y, float yaw);
    int32_t MoveToIncrementPosition(float x, float y, float yaw);
};
}

#endif //FAKESDK_OBSTACLES_AVOID_CLIENT_HPP
