#include "Explore.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace robo::sdk::examples;

static constexpr float EXPLORE_SPEED = 0.4f;
static constexpr float TURN_GAIN = 1.0f;
// a frontier closer than this would be explored without moving
static constexpr float MIN_GOAL_DISTANCE = 1.0f;
static constexpr float GOAL_REACHED = 0.5f;
// a goal not reached in this time counts as unreachable (s)
static constexpr float GOAL_TIMEOUT = 30.0f;
// frontiers this close to an unreachable goal are skipped as well
static constexpr float FAILED_GOAL_RADIUS = 1.0f;
// a route is followed by steering toward the point this far along it
static constexpr float PATH_LOOKAHEAD = 0.8f;
// closer than this to an obstacle the robot only turns, it doesn't walk forward
static constexpr float MIN_CLEARANCE = 0.35f;
// before the first height map arrives the robot turns on the spot to look around
static constexpr float LOOK_AROUND_SPEED = 0.3f;

static float wrapAngle(float a) {
    return std::atan2(std::sin(a), std::cos(a));
}

Explore::Explore(core::Control& control, core::ObstacleMap& map) : roboControl(control), obstacleMap(map) {
}

void Explore::reset() {
    hasGoal = false;
    failedGoals.clear();
}

bool Explore::step(float& x, float& z) {
    x = 0;
    z = 0;
    if (!obstacleMap.available()) {
        z = LOOK_AROUND_SPEED;
        return true;
    }

    const float rx = roboControl.positionX();
    const float ry = roboControl.positionY();
    const float yaw = roboControl.yaw();
    const auto now = std::chrono::steady_clock::now();

    if (hasGoal && std::hypot(goalX - rx, goalY - ry) < GOAL_REACHED) {
        hasGoal = false;
    }
    if (hasGoal && std::chrono::duration<float>(now - goalStart).count() > GOAL_TIMEOUT) {
        std::cout << "[Explore] goal not reached in time, skipping it" << std::endl;
        failedGoals.emplace_back(goalX, goalY);
        hasGoal = false;
    }

    if (!hasGoal) {
        if (!obstacleMap.findFrontier(rx, ry, MIN_GOAL_DISTANCE, failedGoals, FAILED_GOAL_RADIUS, goalX, goalY))
            return false;
        hasGoal = true;
        goalStart = now;
        std::cout << "[Explore] heading to " << goalX << ", " << goalY << std::endl;
    }

    std::vector<std::pair<float, float>> path;
    if (!obstacleMap.planPath(rx, ry, goalX, goalY, GOAL_REACHED, path) || path.size() < 2) {
        failedGoals.emplace_back(goalX, goalY);
        hasGoal = false;
        return true;
    }

    auto aim = path.back();
    for (const auto& point : path) {
        if (std::hypot(point.first - rx, point.second - ry) >= PATH_LOOKAHEAD) {
            aim = point;
            break;
        }
    }
    const float heading = wrapAngle(std::atan2(aim.second - ry, aim.first - rx) - yaw);
    const float aimDistance = std::hypot(aim.first - rx, aim.second - ry);
    const float clearance = obstacleMap.freeDistance(rx, ry, yaw + heading, std::max(aimDistance, MIN_CLEARANCE));

    // turn toward the route first, walk once roughly facing it; slow down in front of obstacles
    z = std::clamp(TURN_GAIN * heading, -0.6f, +0.6f);
    x = EXPLORE_SPEED * std::max(0.0f, std::cos(heading));
    if (clearance < aimDistance)
        x = std::min(x, std::max(0.0f, 0.5f * (clearance - MIN_CLEARANCE)));
    return true;
}
