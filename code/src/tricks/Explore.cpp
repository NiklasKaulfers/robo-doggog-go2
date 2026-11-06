#include "Explore.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace robo::sdk::examples;

static constexpr float EXPLORE_SPEED = 0.4f;
static constexpr float TURN_GAIN = 1.0f;
// goals are picked between these distances (m)
static constexpr float MIN_GOAL_DISTANCE = 1.5f;
static constexpr float MAX_GOAL_DISTANCE = 8.0f;
static constexpr float GOAL_REACHED = 0.5f;
// a goal not reached in this time counts as unreachable (s)
static constexpr float GOAL_TIMEOUT = 30.0f;
// spots this close to an unreachable goal are skipped as well, for the last few failures
static constexpr float FAILED_GOAL_RADIUS = 1.0f;
static constexpr size_t MAX_FAILED_GOALS = 20;
// the trail stores a point every TRAIL_SPACING meters, up to MAX_TRAIL points
static constexpr float TRAIL_SPACING = 0.5f;
static constexpr size_t MAX_TRAIL = 400;
// goal choice: distance to the trail counts up to NOVELTY_CAP, edges of the known map
// get a bonus, and a random part keeps the wandering unpredictable
static constexpr float NOVELTY_CAP = 4.0f;
static constexpr float FRONTIER_BONUS = 2.0f;
static constexpr float RANDOMNESS = 2.0f;
// a route is followed by steering toward the point this far along it
static constexpr float PATH_LOOKAHEAD = 0.8f;
// closer than this to an obstacle the robot only turns, it doesn't walk forward
static constexpr float MIN_CLEARANCE = 0.35f;
// turning on the spot to look around: after each goal and while nothing is known
static constexpr float LOOK_AROUND_SPEED = 0.3f;
static constexpr auto LOOK_AROUND_TIME = std::chrono::milliseconds(2000);
static constexpr auto LOG_INTERVAL = std::chrono::seconds(3);

static float wrapAngle(float a) {
    return std::atan2(std::sin(a), std::cos(a));
}

Explore::Explore(core::Control& control, core::ObstacleMap& map) : roboControl(control), obstacleMap(map) {
}

void Explore::reset() {
    hasGoal = false;
    failedGoals.clear();
    trail.clear();
    lookAroundUntil = {};
}

void Explore::log(const std::string& message) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastLog < LOG_INTERVAL) return;
    lastLog = now;
    std::cout << "[Explore] " << message << std::endl;
}

bool Explore::chooseGoal(float rx, float ry) {
    std::vector<core::ObstacleMap::Spot> spots;
    obstacleMap.reachableSpots(rx, ry, MIN_GOAL_DISTANCE, MAX_GOAL_DISTANCE, spots);
    if (spots.empty()) {
        // outside the map means the height map and the robot's odometry use different frames
        log("no reachable spots around " + std::to_string(rx) + ", " + std::to_string(ry) +
            (obstacleMap.contains(rx, ry) ? " (blocked in or nothing seen yet)" : " (robot position is outside the height map)"));
        return false;
    }

    std::uniform_real_distribution<float> random(0.0f, RANDOMNESS);
    float bestScore = -1;
    for (const auto& spot : spots) {
        const bool failed = std::any_of(failedGoals.begin(), failedGoals.end(), [&](const auto& p) {
            return std::hypot(p.first - spot.x, p.second - spot.y) < FAILED_GOAL_RADIUS;
        });
        if (failed) continue;

        float novelty = NOVELTY_CAP;
        for (const auto& p : trail)
            novelty = std::min(novelty, std::hypot(p.first - spot.x, p.second - spot.y));

        const float score = novelty + (spot.frontier ? FRONTIER_BONUS : 0.0f) + random(rng);
        if (score > bestScore) {
            bestScore = score;
            goalX = spot.x;
            goalY = spot.y;
        }
    }
    return bestScore >= 0;
}

void Explore::step(float& x, float& z) {
    x = 0;
    z = 0;
    const auto now = std::chrono::steady_clock::now();

    if (!obstacleMap.available()) {
        log("no height map received yet, looking around");
        z = lookAroundDirection * LOOK_AROUND_SPEED;
        return;
    }
    if (now < lookAroundUntil) {
        z = lookAroundDirection * LOOK_AROUND_SPEED;
        return;
    }

    const float rx = roboControl.positionX();
    const float ry = roboControl.positionY();
    const float yaw = roboControl.yaw();

    if (trail.empty() || std::hypot(trail.back().first - rx, trail.back().second - ry) >= TRAIL_SPACING) {
        trail.emplace_back(rx, ry);
        if (trail.size() > MAX_TRAIL) trail.pop_front();
    }

    const auto markFailed = [&]() {
        failedGoals.emplace_back(goalX, goalY);
        if (failedGoals.size() > MAX_FAILED_GOALS) failedGoals.pop_front();
        hasGoal = false;
    };

    if (hasGoal && std::hypot(goalX - rx, goalY - ry) < GOAL_REACHED) {
        // arrived: have a short look around, then walk on
        hasGoal = false;
        lookAroundDirection = std::uniform_int_distribution<int>(0, 1)(rng) ? 1.0f : -1.0f;
        lookAroundUntil = now + LOOK_AROUND_TIME;
        return;
    }
    if (hasGoal && std::chrono::duration<float>(now - goalStart).count() > GOAL_TIMEOUT) {
        std::cout << "[Explore] goal not reached in time, picking another one" << std::endl;
        markFailed();
    }

    if (!hasGoal) {
        if (!chooseGoal(rx, ry)) {
            // nowhere to go right now (e.g. boxed in): look around to see more
            failedGoals.clear();
            z = lookAroundDirection * LOOK_AROUND_SPEED;
            return;
        }
        hasGoal = true;
        goalStart = now;
        std::cout << "[Explore] walking to " << goalX << ", " << goalY << std::endl;
    }

    std::vector<std::pair<float, float>> path;
    if (!obstacleMap.planPath(rx, ry, goalX, goalY, GOAL_REACHED, path) || path.size() < 2) {
        log("no route to the goal, picking another one");
        markFailed();
        return;
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
    if (clearance < aimDistance) {
        x = std::min(x, std::max(0.0f, 0.5f * (clearance - MIN_CLEARANCE)));
        if (x == 0) log("obstacle " + std::to_string(clearance) + "m ahead, only turning");
    }
}
