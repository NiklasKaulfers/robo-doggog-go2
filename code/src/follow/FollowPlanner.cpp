#include "FollowPlanner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "../tracking/CameraGeometry.h"

using namespace robo::sdk::follow;
using robo::sdk::tracking::CameraGeometry;
using robo::sdk::tracking::RobotPose;
using robo::sdk::tracking::TargetTracker;

// --- following ---
static constexpr float TARGET_DISTANCE = 0.5f;
// speed caps while following (m/s, rad/s)
static constexpr float MAX_FORWARD_SPEED = 0.6f;
static constexpr float MAX_YAW_SPEED = 0.6f;
static constexpr float DISTANCE_GAIN = 0.5f;
static constexpr float TURN_GAIN = 1.0f;
// route planning: how far ahead the path to the person must be free
static constexpr float PLAN_LOOKAHEAD = 1.5f;
// the person's legs show up as an obstacle in the height map, stop checking before them
static constexpr float PERSON_MARGIN = 0.4f;
// closer than this to an obstacle the robot only turns, it doesn't walk forward
static constexpr float MIN_CLEARANCE = 0.35f;
// detours are tried in steps of ~10 degrees up to ~70 degrees to either side
static constexpr float DETOUR_STEP = 0.17f;
static constexpr int DETOUR_STEPS = 7;
// a planned route is followed by steering toward the point this far along it
static constexpr float PATH_LOOKAHEAD = 0.8f;
// a height map older than this (s) isn't trusted for obstacles, the robot only turns
static constexpr float MAP_STALE_SECONDS = 1.0f;

// --- searching (starting values, tune on the robot) ---
// Pursue: close enough to the predicted position, or given up on reaching it (s)
static constexpr float ARRIVAL_DISTANCE = 0.6f;
static constexpr float PURSUE_TIMEOUT = 15.0f;
// LookAround: turn speed (rad/s) and how far to turn (rad, one full turn)
static constexpr float LOOK_AROUND_SPEED = 0.4f;
static constexpr float LOOK_AROUND_ANGLE = 2 * static_cast<float>(M_PI);
// Sweep: the search area around where they were last seen starts at this radius (m),
// grows by this much per second up to the maximum (reached after ~2 min of sweeping).
// Much more than 20 m doesn't help: the remembered height map is 40 x 40 m around the
// robot and forgets terrain after 3 minutes
static constexpr float SWEEP_START_RADIUS = 3.0f;
static constexpr float SWEEP_GROWTH = 0.15f;
static constexpr float SWEEP_MAX_RADIUS = 20.0f;
// goals in the direction they walked score up to this much more (the ExplorePlanner's
// novelty counts up to 4, its randomness up to 2), when they walked at least this fast (m/s)
static constexpr float SWEEP_DIRECTION_BONUS = 3.0f;
static constexpr float SWEEP_MIN_WALKING_SPEED = 0.2f;
// the area size is logged whenever it grew by this much (m)
static constexpr float SWEEP_LOG_STEP = 5.0f;
// total time of Pursue + LookAround + Sweep before giving up and standing still (s)
static constexpr float SEARCH_LIMIT = 600.0f;
// one step counts for the search timers at most this much (s), so long gaps between
// steps (remote controller, sensor trouble) don't use up the search time
static constexpr float MAX_STEP_SECONDS = 0.5f;

static float seconds(FollowPlanner::Clock::duration d) {
    return std::chrono::duration<float>(d).count();
}

static std::string num(float value, int digits = 1) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.*f", digits, value);
    return buffer;
}

FollowPlanner::FollowPlanner(core::ObstacleMap& obstacleMap, uint32_t seed)
    : map(obstacleMap), sweep(obstacleMap, seed, [this] { return activeSeconds; }) {
    logger = [](const std::string& message, bool warning) {
        (warning ? std::cerr : std::cout) << "[Follow] " << message << std::endl;
    };
    sweep.setLogger([this](const std::string& message) { say("sweep: " + message); });
}

bool FollowPlanner::init(const std::string& resourceDir) {
    return tracker.init(resourceDir);
}

bool FollowPlanner::loadReferencePhotos(const std::string& folder) {
    return tracker.loadReferencePhotos(folder);
}

void FollowPlanner::setLogger(Logger newLogger) {
    logger = std::move(newLogger);
}

void FollowPlanner::say(const std::string& message, bool warning) {
    if (logger) logger(message, warning);
}

const char* FollowPlanner::phaseName(Phase phase) {
    switch (phase) {
        case Phase::WaitingForTarget: return "WaitingForTarget";
        case Phase::Following: return "Following";
        case Phase::Pursue: return "Pursue";
        case Phase::LookAround: return "LookAround";
        case Phase::Sweep: return "Sweep";
        case Phase::GaveUp: return "GaveUp";
    }
    return "?";
}

void FollowPlanner::enter(Phase phase, const std::string& reason) {
    if (phase == current.phase) return;
    say(std::string(phaseName(current.phase)) + " -> " + phaseName(phase) + ": " + reason);
    current.phase = phase;
}

void FollowPlanner::pause() {
    paused = true;
}

void FollowPlanner::reset() {
    tracker.dropTarget();
    sweep.reset();
    enter(Phase::WaitingForTarget, "operator reset, the next person becomes the target");
    current.searchSeconds = 0;
}

const FollowPlanner::Step& FollowPlanner::step(const std::vector<ml::PersonDetection>& detections, const cv::Mat& frame,
                                               const RobotPose& pose, Clock::time_point now) {
    // time and turning since the last step; nothing after a pause, at most MAX_STEP_SECONDS
    const bool counted = haveStep && !paused;
    const float dt = counted ? std::clamp(seconds(now - lastTime), 0.0f, MAX_STEP_SECONDS) : 0.0f;
    const float turned = counted ? CameraGeometry::wrapAngle(pose.yaw - lastYaw) : 0.0f;
    haveStep = true;
    paused = false;
    lastTime = now;
    lastYaw = pose.yaw;
    activeSeconds += dt;

    current.target = tracker.update(detections, frame, pose, now);
    const auto& target = current.target;
    current.x = current.z = 0;

    if (target.state == TargetTracker::State::Searching) {
        enter(Phase::WaitingForTarget, "no target");
        current.searchSeconds = 0;
        return current;
    }

    if (target.state == TargetTracker::State::Lost && !target.visible) {
        search(pose, dt, turned);
        return current;
    }

    // in view, or out of view only for a few frames
    if (current.phase != Phase::Following) {
        const bool searching = current.phase != Phase::WaitingForTarget;
        std::string reason = searching ? "found the target again after " + num(current.searchSeconds, 0) + " s"
                                       : "found the target";
        if (!target.foundBy.empty()) reason += " (" + target.foundBy + ")";
        enter(Phase::Following, reason);
    }
    current.searchSeconds = 0;
    if (target.visible) {
        chase(CameraGeometry::distance(target.target.box, frame.cols), CameraGeometry::angle(target.target.box, frame.cols), pose);
    } else {
        // out of view for a few frames: head where they should be
        chase(target.predictedDistance, target.predictedAngle, pose);
    }
    return current;
}

void FollowPlanner::search(const RobotPose& pose, float dt, float turned) {
    const auto& target = current.target;
    const Phase before = current.phase;
    if (before == Phase::Following || before == Phase::WaitingForTarget) {
        current.searchSeconds = 0;
        phaseSeconds = 0;
        lookDirection = target.lostSide >= 0 ? 1.0f : -1.0f;
        enter(Phase::Pursue, "lost the target, heading where they were going (" + num(target.predictedDistance) + " m, " +
                             num(target.predictedAngle * 180 / static_cast<float>(M_PI), 0) + " deg)");
    } else {
        current.searchSeconds += dt;
        phaseSeconds += dt;
    }

    if (current.phase != Phase::GaveUp && current.searchSeconds > SEARCH_LIMIT) {
        enter(Phase::GaveUp, "searched for " + num(current.searchSeconds, 0) + " s without finding them, standing still and watching");
    }

    if (current.phase == Phase::Pursue) {
        if (target.predictedDistance < ARRIVAL_DISTANCE) {
            startLookAround("arrived where they were heading");
        } else if (phaseSeconds > PURSUE_TIMEOUT) {
            startLookAround("could not get where they were heading in " + num(PURSUE_TIMEOUT, 0) + " s");
        } else {
            chase(target.predictedDistance, target.predictedAngle, pose);
            return;
        }
    }

    if (current.phase == Phase::LookAround) {
        // measured by how far the robot really turned, not by time: turning slower or
        // being held up doesn't end it early, turning faster doesn't make it spin on
        if (before == Phase::LookAround) lookTurned += lookDirection * turned;
        if (lookTurned >= LOOK_AROUND_ANGLE) {
            startSweep(pose);
        } else {
            current.z = lookDirection * LOOK_AROUND_SPEED;
            return;
        }
    }

    if (current.phase == Phase::Sweep) {
        sweepArea.radius = std::min(SWEEP_START_RADIUS + SWEEP_GROWTH * phaseSeconds, SWEEP_MAX_RADIUS);
        if (sweepArea.radius >= sweepLoggedRadius + SWEEP_LOG_STEP) {
            sweepLoggedRadius = sweepArea.radius;
            say("sweep area grew to " + num(sweepArea.radius) + " m");
        }
        sweep.setSearchFocus(sweepArea);
        const core::ExploreCommand command = sweep.step({pose.x, pose.y, pose.yaw});
        current.x = command.x;
        current.z = command.z;
    }
    // GaveUp: stand still, the tracker keeps watching
}

void FollowPlanner::startLookAround(const std::string& reason) {
    phaseSeconds = 0;
    lookTurned = 0;
    enter(Phase::LookAround, reason + ", looking around to the " + (lookDirection > 0 ? "left" : "right"));
}

void FollowPlanner::startSweep(const RobotPose& pose) {
    const auto& target = current.target;
    phaseSeconds = 0;
    sweep.reset();
    sweepArea = {};
    // where they were last seen; without that (shouldn't happen once a target was
    // followed) around the robot rather than around the odometry origin
    sweepArea.x = target.positionKnown ? target.lastSeenX : pose.x;
    sweepArea.y = target.positionKnown ? target.lastSeenY : pose.y;
    sweepArea.radius = SWEEP_START_RADIUS;
    sweepLoggedRadius = SWEEP_START_RADIUS;
    std::string reason = "looked around, sweeping within " + num(SWEEP_START_RADIUS) + " m of where they were last seen";
    if (std::hypot(target.walkingX, target.walkingY) >= SWEEP_MIN_WALKING_SPEED) {
        sweepArea.direction = std::atan2(target.walkingY, target.walkingX);
        sweepArea.directionBonus = SWEEP_DIRECTION_BONUS;
        reason += ", mostly the way they walked";
    }
    enter(Phase::Sweep, reason);
}

void FollowPlanner::chase(float distance, float angle, const RobotPose& pose) {
    // proportional control so the motion is smooth instead of walking straight and
    // suddenly turning once the person leaves a band around the image center
    // (x: move toward the person if too far, away if too close)
    float x = std::clamp(DISTANCE_GAIN * (distance - TARGET_DISTANCE), -MAX_FORWARD_SPEED, MAX_FORWARD_SPEED);

    // a height map that stopped updating is treated as no map: its obstacles can't be
    // trusted, so don't walk forward on it. INFINITY = never received
    const float mapAge = map.secondsSinceUpdate();
    const bool mapOld = std::isfinite(mapAge) && mapAge > MAP_STALE_SECONDS;
    if (mapOld != mapStale) {
        mapStale = mapOld;
        say(mapStale ? "height map is stale, only turning" : "height map is updating again", mapStale);
    }

    // when walking toward the person, take a detour around obstacles in the way
    float heading = angle;
    if (x > 0 && mapStale) {
        x = 0;
    } else if (x > 0) {
        float clearance = 0;
        heading = CameraGeometry::wrapAngle(planHeading(angle, distance, pose, clearance));
        // x drives the body along its current yaw, not along the planned heading, so the
        // room straight ahead limits it as well. That ray ends PERSON_MARGIN before the
        // person (same as planHeading), so their legs don't count as an obstacle
        const float needed = std::min(distance - PERSON_MARGIN, PLAN_LOOKAHEAD);
        if (map.available() && needed > 0) {
            const float ahead = map.freeDistance(pose.x, pose.y, pose.yaw, needed);
            clearance = std::min(clearance, ahead >= needed ? PLAN_LOOKAHEAD : ahead);
        }
        // slow down in front of obstacles and only turn if nothing is free
        x = std::min(x, std::max(0.0f, 0.5f * (clearance - MIN_CLEARANCE)));
        // turn onto the detour before walking along it
        x *= std::max(0.0f, std::cos(heading));
    }

    // rotate toward the chosen heading, proportional to how far off it is
    current.x = x;
    current.z = std::clamp(TURN_GAIN * heading, -MAX_YAW_SPEED, MAX_YAW_SPEED);
}

float FollowPlanner::planHeading(float personAngle, float distance, const RobotPose& pose, float& clearance) {
    const float needed = std::min(distance - PERSON_MARGIN, PLAN_LOOKAHEAD);
    if (!map.available() || needed <= 0) {
        clearance = PLAN_LOOKAHEAD;
        return personAngle;
    }

    const float x = pose.x, y = pose.y, yaw = pose.yaw;

    // route through everything seen in the last minutes, so the robot can use a way
    // around that it saw earlier but that is out of the LiDAR's range now
    const float targetX = x + distance * std::cos(yaw + personAngle);
    const float targetY = y + distance * std::sin(yaw + personAngle);
    std::vector<std::pair<float, float>> path;
    if (map.planPath(x, y, targetX, targetY, PERSON_MARGIN, path) && path.size() > 1) {
        auto aim = path.back();
        for (const auto& point : path) {
            if (std::hypot(point.first - x, point.second - y) >= PATH_LOOKAHEAD) {
                aim = point;
                break;
            }
        }
        const float heading = CameraGeometry::wrapAngle(std::atan2(aim.second - y, aim.first - x) - yaw);
        const float aimDistance = std::hypot(aim.first - x, aim.second - y);
        clearance = map.freeDistance(x, y, yaw + heading, std::max(std::min(needed, aimDistance), MIN_CLEARANCE));
        // the route already goes around everything, only nearby obstacles slow it down
        if (clearance >= std::min(needed, aimDistance)) clearance = PLAN_LOOKAHEAD;
        return heading;
    }

    // no route found (e.g. target outside the map): detour by looking straight ahead

    // try the direct heading first, then detours further and further to the side.
    // the side of the last detour is tried first so the robot doesn't zig-zag
    float bestHeading = personAngle;
    float bestFree = -1;
    for (int i = 0; i <= 2 * DETOUR_STEPS; i++) {
        const int k = (i + 1) / 2 * (i % 2 ? lastDetourSide : -lastDetourSide);
        const float heading = CameraGeometry::wrapAngle(personAngle + k * DETOUR_STEP);
        const float free = map.freeDistance(x, y, yaw + heading, needed);

        if (free >= needed) {
            if (k != 0) lastDetourSide = k > 0 ? 1 : -1;
            clearance = free;
            return heading;
        }
        if (free > bestFree) {
            bestFree = free;
            bestHeading = heading;
        }
    }

    // nothing is completely free: head where there is the most room
    clearance = bestFree;
    return bestHeading;
}

void FollowPlanner::drawDebug(cv::Mat& image) const {
    if (image.empty()) return;
    tracker.drawDebug(image);
    char text[96];
    std::snprintf(text, sizeof text, "%s  search %.0fs", phaseName(current.phase), current.searchSeconds);
    cv::putText(image, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
}
