#ifndef FOLLOW_EXPLORE_H
#define FOLLOW_EXPLORE_H

#include "../core/Control.h"
#include "../core/ObstacleMap.h"
#include <chrono>
#include <utility>
#include <vector>

namespace robo::sdk::examples {
// Explores the surroundings: repeatedly walks to the nearest edge of the known map
// until every reachable area has been seen.
class Explore {

    core::Control& roboControl;
    core::ObstacleMap& obstacleMap;

    bool hasGoal = false;
    float goalX = 0, goalY = 0;
    std::chrono::steady_clock::time_point goalStart;
    // goals that couldn't be reached, they are skipped from now on
    std::vector<std::pair<float, float>> failedGoals;

public:

    Explore(core::Control& control, core::ObstacleMap& map);

    // forgets goals from a previous exploration
    void reset();
    // movement for this step (x: forward speed, z: yaw speed);
    // false once there is nothing left to explore
    bool step(float& x, float& z);
};
}

#endif //FOLLOW_EXPLORE_H
