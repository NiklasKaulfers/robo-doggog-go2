#include "ExplorePlanner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

// hard limits, not configurable
static constexpr float MAX_FORWARD_SPEED = 0.4f;
static constexpr float MAX_YAW_SPEED = 0.6f;
// beyond this heading error (rad) the turn direction is kept from the previous step, so
// a target right behind the robot doesn't make it flip between left and right
static constexpr float TURN_HOLD_HEADING = 2.8f;

namespace robo::sdk::core
{
    ExplorePlanner::ExplorePlanner(ObstacleMap& obstacleMap, uint32_t seed, Clock newClock, ExploreConfig config)
        : map(obstacleMap), clock(std::move(newClock)), cfg(config), rng(seed)
    {
        if (!clock) {
            const auto start = std::chrono::steady_clock::now();
            clock = [start] { return std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count(); };
        }
        cfg.exploreSpeed = std::clamp(cfg.exploreSpeed, 0.0f, MAX_FORWARD_SPEED);
        cfg.lookAroundSpeed = std::clamp(cfg.lookAroundSpeed, 0.0f, MAX_YAW_SPEED);
        logger = [](const std::string& message) { std::cout << "[Explore] " << message << std::endl; };
    }

    void ExplorePlanner::setLogger(Logger newLogger)
    {
        logger = std::move(newLogger);
    }

    void ExplorePlanner::reset()
    {
        currentState = State::NoMap;
        hasGoal_ = false;
        goalElapsed_ = 0;
        failedGoals_.clear();
        trail_.clear();
        lookAroundUntil = -1e9f;
        lookAroundDirection = 1;
        lastLog = -1e9f;
        mapStale = false;
        frameMismatch = false;
        planFailures = 0;
        walkedLastStep = false;
        lastTurnSign = 0;
        lastHadSpots = false;
    }

    float ExplorePlanner::wrapAngle(float a)
    {
        return std::atan2(std::sin(a), std::cos(a));
    }

    void ExplorePlanner::say(const std::string& message)
    {
        if (logger) logger(message);
    }

    void ExplorePlanner::log(const std::string& message, float now)
    {
        if (now - lastLog < cfg.logInterval) return;
        lastLog = now;
        say(message);
    }

    void ExplorePlanner::startLookAround(float now)
    {
        lookAroundDirection = std::uniform_int_distribution<int>(0, 1)(rng) ? 1.0f : -1.0f;
        lookAroundUntil = now + cfg.lookAroundTime;
    }

    void ExplorePlanner::markFailed()
    {
        failedGoals_.emplace_back(goalX, goalY);
        while (failedGoals_.size() > cfg.maxFailedGoals) failedGoals_.pop_front();
        hasGoal_ = false;
    }

    float ExplorePlanner::scoreSpot(const ObstacleMap::Spot& spot, float randomPart) const
    {
        float novelty = cfg.noveltyCap;
        for (const auto& p : trail_)
            novelty = std::min(novelty, std::hypot(p.first - spot.x, p.second - spot.y));
        return novelty + (spot.frontier ? cfg.frontierBonus : 0.0f) + randomPart;
    }

    bool ExplorePlanner::chooseGoal(float rx, float ry)
    {
        std::vector<ObstacleMap::Spot> spots;
        map.reachableSpots(rx, ry, cfg.minGoalDistance, cfg.maxGoalDistance, spots);
        lastHadSpots = !spots.empty();
        if (spots.empty()) {
            // outside the map means the height map and the robot's odometry use different frames
            log("no reachable spots around " + std::to_string(rx) + ", " + std::to_string(ry) +
                (map.contains(rx, ry) ? " (blocked in or nothing seen yet)" : " (robot position is outside the height map)"),
                clock());
            return false;
        }

        std::uniform_real_distribution<float> random(0.0f, cfg.randomness);
        float bestScore = -1;
        for (const auto& spot : spots) {
            const bool failed = std::any_of(failedGoals_.begin(), failedGoals_.end(), [&](const auto& p) {
                return std::hypot(p.first - spot.x, p.second - spot.y) < cfg.failedGoalRadius;
            });
            if (failed) continue;

            const float score = scoreSpot(spot, random(rng));
            if (score > bestScore) {
                bestScore = score;
                goalX = spot.x;
                goalY = spot.y;
            }
        }
        return bestScore >= 0;
    }

    ExploreCommand ExplorePlanner::step(const ExplorePose& pose)
    {
        ExploreCommand cmd = stepImpl(pose);
        // last line of defense: whatever happened above, the output stays in the safe range
        if (!std::isfinite(cmd.x)) cmd.x = 0;
        if (!std::isfinite(cmd.z)) cmd.z = 0;
        cmd.x = std::clamp(cmd.x, 0.0f, cfg.exploreSpeed);
        cmd.z = std::clamp(cmd.z, -MAX_YAW_SPEED, MAX_YAW_SPEED);
        return cmd;
    }

    ExploreCommand ExplorePlanner::stepImpl(const ExplorePose& pose)
    {
        ExploreCommand cmd;
        const float now = clock();
        const bool continuing = walkedLastStep;
        const float sinceLastStep = now - lastStepTime;
        walkedLastStep = false;
        lastStepTime = now;

        if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.yaw)) {
            currentState = State::InvalidPose;
            log("robot pose is not valid, stopping", now);
            return cmd;
        }

        if (!map.available()) {
            currentState = State::NoMap;
            log("no height map received yet, looking around", now);
            cmd.z = lookAroundDirection * cfg.lookAroundSpeed;
            return cmd;
        }

        // the height map and odometry must use the same frame, or every route is wrong
        const bool mismatch = !map.checkFrame(pose.x, pose.y);
        if (mismatch != frameMismatch) {
            frameMismatch = mismatch;
            say(mismatch ? "height map and robot position are in different frames, stopping"
                         : "height map and robot position match again");
        }
        if (mismatch) {
            currentState = State::FrameMismatch;
            return cmd;
        }

        // the map was received but stopped updating: exploring relies on it entirely, so stand
        // still until it comes back. INFINITY = never received, handled above
        const float mapAge = map.secondsSinceUpdate();
        const bool stale = std::isfinite(mapAge) && mapAge > cfg.mapStaleSeconds;
        if (stale != mapStale) {
            mapStale = stale;
            say(stale ? "height map is stale, stopping" : "height map is updating again");
        }
        if (stale) {
            currentState = State::MapStale;
            return cmd;
        }

        if (now < lookAroundUntil) {
            currentState = State::LookingAround;
            cmd.z = lookAroundDirection * cfg.lookAroundSpeed;
            return cmd;
        }

        const float rx = pose.x, ry = pose.y, yaw = pose.yaw;

        if (trail_.empty() || std::hypot(trail_.back().first - rx, trail_.back().second - ry) >= cfg.trailSpacing) {
            trail_.emplace_back(rx, ry);
            while (trail_.size() > cfg.maxTrail) trail_.pop_front();
        }

        const auto arrived = [&]() {
            // have a short look around, then walk on
            hasGoal_ = false;
            planFailures = 0;
            startLookAround(now);
            currentState = State::LookingAround;
            cmd.z = lookAroundDirection * cfg.lookAroundSpeed;
        };

        if (hasGoal_ && std::hypot(goalX - rx, goalY - ry) < cfg.goalReached) {
            arrived();
            return cmd;
        }

        if (hasGoal_) {
            // only time spent walking counts, and a long gap between steps counts as one short step
            goalElapsed_ += continuing ? std::clamp(sinceLastStep, 0.0f, cfg.maxTickGap) : 0.0f;
            if (goalElapsed_ > cfg.goalTimeout) {
                say("goal not reached in time, picking another one");
                markFailed();
            }
        }

        if (!hasGoal_) {
            if (!chooseGoal(rx, ry)) {
                // nowhere to go right now (e.g. boxed in): look around to see more. If spots
                // exist but all are near failed goals, forget the oldest half of those
                if (lastHadSpots && !failedGoals_.empty())
                    failedGoals_.erase(failedGoals_.begin(), failedGoals_.begin() + (failedGoals_.size() + 1) / 2);
                startLookAround(now);
                currentState = State::LookingAround;
                cmd.z = lookAroundDirection * cfg.lookAroundSpeed;
                return cmd;
            }
            hasGoal_ = true;
            goalElapsed_ = 0;
            say("walking to " + std::to_string(goalX) + ", " + std::to_string(goalY));
        }

        std::vector<std::pair<float, float>> path;
        if (!map.planPath(rx, ry, goalX, goalY, cfg.goalReached, path)) {
            log("no route to the goal, picking another one", now);
            markFailed();
            currentState = State::Searching;
            if (++planFailures >= cfg.maxPlanFailures) {
                planFailures = 0;
                startLookAround(now);
                currentState = State::LookingAround;
                cmd.z = lookAroundDirection * cfg.lookAroundSpeed;
            }
            return cmd;
        }
        planFailures = 0;
        if (path.size() < 2) {
            // the route is just the start cell: effectively there already
            arrived();
            return cmd;
        }

        currentState = State::Walking;
        walkedLastStep = true;

        auto aim = path.back();
        for (const auto& point : path) {
            if (std::hypot(point.first - rx, point.second - ry) >= cfg.pathLookahead) {
                aim = point;
                break;
            }
        }
        const float heading = wrapAngle(std::atan2(aim.second - ry, aim.first - rx) - yaw);
        const float aimDistance = std::hypot(aim.first - rx, aim.second - ry);
        // x moves the body along its current yaw, not along the route: the room straight
        // ahead limits the speed as well as the room along the route
        const float lookDistance = std::max(aimDistance, cfg.minClearance);
        const float clearance = std::min(map.freeDistance(rx, ry, yaw + heading, lookDistance),
                                         map.freeDistance(rx, ry, yaw, lookDistance));

        // turn toward the route first, walk once roughly facing it
        float turn = std::clamp(cfg.turnGain * heading, -MAX_YAW_SPEED, MAX_YAW_SPEED);
        if (std::abs(heading) > TURN_HOLD_HEADING && continuing && lastTurnSign != 0)
            turn = lastTurnSign * std::abs(turn);
        lastTurnSign = turn > 0 ? 1.0f : (turn < 0 ? -1.0f : 0.0f);
        cmd.z = turn;

        cmd.x = cfg.exploreSpeed * std::max(0.0f, std::cos(heading));
        cmd.x = std::min(cmd.x, aimDistance);
        // slow down in front of obstacles, also when only the far end of the look is blocked
        if (clearance < lookDistance - 1e-3f) {
            cmd.x = std::min(cmd.x, std::max(0.0f, 0.5f * (clearance - cfg.minClearance)));
            if (cmd.x == 0) log("obstacle " + std::to_string(clearance) + "m ahead, only turning", now);
        }
        return cmd;
    }
}
