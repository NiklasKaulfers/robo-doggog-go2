#ifndef FOLLOW_EXPLORE_H
#define FOLLOW_EXPLORE_H

#include "../core/Control.h"
#include "../core/ExplorePlanner.h"
#include "../core/ObstacleMap.h"

namespace robo::sdk::examples {
// Roams around the place: keeps walking to new spots, preferring areas it hasn't been
// to lately and the edges of what it has seen, with some randomness so it wanders
// freely instead of following a fixed pattern. Runs until it is switched off.
// The decisions are made by core::ExplorePlanner, this only connects it to the robot.
class Explore {

    core::Control& roboControl;
    core::ExplorePlanner planner;

public:

    Explore(core::Control& control, core::ObstacleMap& map);

    // forgets goals and the trail from a previous run
    void reset();
    // movement for this step (x: forward speed, z: yaw speed)
    void step(float& x, float& z);
};
}

#endif //FOLLOW_EXPLORE_H
