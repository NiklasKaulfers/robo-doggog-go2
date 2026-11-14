#include "Explore.h"

#include <random>

using namespace robo::sdk::examples;

Explore::Explore(core::Control& control, core::ObstacleMap& map)
    : roboControl(control), planner(map, std::random_device {}()) {
}

void Explore::reset() {
    planner.reset();
}

void Explore::step(float& x, float& z) {
    const core::ExploreCommand command = planner.step({roboControl.positionX(), roboControl.positionY(), roboControl.yaw()});
    x = command.x;
    z = command.z;
}
