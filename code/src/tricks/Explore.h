#ifndef FOLLOW_EXPLORE_H
#define FOLLOW_EXPLORE_H

#include "../core/Control.h"
#include "../core/ObstacleMap.h"
#include <chrono>
#include <deque>
#include <random>
#include <utility>
#include <vector>

namespace robo::sdk::examples {
// Roams around the place: keeps walking to new spots, preferring areas it hasn't been
// to lately and the edges of what it has seen, with some randomness so it wanders
// freely instead of following a fixed pattern. Runs until it is switched off.
class Explore {

    core::Control& roboControl;
    core::ObstacleMap& obstacleMap;

    bool hasGoal = false;
    float goalX = 0, goalY = 0;
    std::chrono::steady_clock::time_point goalStart;
    // goals that couldn't be reached recently, they are skipped for a while
    std::deque<std::pair<float, float>> failedGoals;
    // where the robot walked recently, so it goes somewhere new
    std::deque<std::pair<float, float>> trail;
    // after reaching a goal the robot looks around a moment before walking on
    std::chrono::steady_clock::time_point lookAroundUntil;
    float lookAroundDirection = 1;
    std::mt19937 rng {std::random_device {}()};

public:

    Explore(core::Control& control, core::ObstacleMap& map);

    // forgets goals and the trail from a previous run
    void reset();
    // movement for this step (x: forward speed, z: yaw speed)
    void step(float& x, float& z);

private:

    bool chooseGoal(float rx, float ry);
};
}

#endif //FOLLOW_EXPLORE_H
