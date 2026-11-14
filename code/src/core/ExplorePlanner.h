#ifndef FOLLOW_EXPLOREPLANNER_H
#define FOLLOW_EXPLOREPLANNER_H

#include "ObstacleMap.h"
#include <cstdint>
#include <deque>
#include <functional>
#include <random>
#include <string>
#include <utility>

namespace robo::sdk::core
{
// tuning of the exploring behavior. Speeds are capped in the planner's constructor:
// forward speed never exceeds 0.4 m/s and turning never 0.6 rad/s, whatever is set here
struct ExploreConfig
{
    float exploreSpeed = 0.4f;
    float turnGain = 1.0f;
    // goals are picked between these distances (m)
    float minGoalDistance = 1.5f;
    float maxGoalDistance = 8.0f;
    float goalReached = 0.5f;
    // a goal not reached after walking this long counts as unreachable (s)
    float goalTimeout = 30.0f;
    // spots this close to an unreachable goal are skipped as well, for the last few failures
    float failedGoalRadius = 1.0f;
    size_t maxFailedGoals = 20;
    // the trail stores a point every trailSpacing meters, up to maxTrail points
    float trailSpacing = 0.5f;
    size_t maxTrail = 400;
    // goal choice: distance to the trail counts up to noveltyCap, edges of the known map
    // get a bonus, and a random part keeps the wandering unpredictable
    float noveltyCap = 4.0f;
    float frontierBonus = 2.0f;
    float randomness = 2.0f;
    // a route is followed by steering toward the point this far along it
    float pathLookahead = 0.8f;
    // closer than this to an obstacle the robot only turns, it doesn't walk forward
    float minClearance = 0.35f;
    // turning on the spot to look around: after each goal and while nothing is known
    float lookAroundSpeed = 0.3f;
    float lookAroundTime = 2.0f;
    float logInterval = 3.0f;
    // a height map older than this (s) is not trusted, the robot stops instead
    float mapStaleSeconds = 1.0f;
    // after this many routes in a row that couldn't be planned the robot looks around
    int maxPlanFailures = 5;
    // the time of one step counts for the goal timeout at most this much (s), so long
    // gaps between steps (remote control, stale map) don't time out the goal
    float maxTickGap = 0.5f;
};

struct ExplorePose
{
    float x = 0, y = 0, yaw = 0;
};

struct ExploreCommand
{
    // forward speed and yaw speed
    float x = 0, z = 0;
};

// Decides where the robot roams: picks goals on the remembered terrain, plans routes to
// them and turns that into forward / yaw speed. Needs only the ObstacleMap, no robot, so
// it can be tested with a simulated one. Not thread safe, call from one thread.
// Output is always within x in [0, 0.4], z in [-0.6, 0.6]
class ExplorePlanner
{
public:
    using Clock = ObstacleMap::Clock;
    using Logger = std::function<void(const std::string&)>;

    enum class State { NoMap, MapStale, FrameMismatch, InvalidPose, LookingAround, Searching, Walking };

private:
    ObstacleMap& map;
    Clock clock;
    ExploreConfig cfg;
    std::mt19937 rng;
    Logger logger;

    State currentState = State::NoMap;
    bool hasGoal_ = false;
    float goalX = 0, goalY = 0;
    // seconds spent walking toward the current goal
    float goalElapsed_ = 0;
    std::deque<std::pair<float, float>> failedGoals_;
    std::deque<std::pair<float, float>> trail_;
    float lookAroundUntil = -1e9f;
    float lookAroundDirection = 1;
    float lastLog = -1e9f;
    bool mapStale = false;
    bool frameMismatch = false;
    int planFailures = 0;
    // whether the previous step reached the walking branch, and when the previous step was
    bool walkedLastStep = false;
    float lastStepTime = 0;
    float lastTurnSign = 0;
    // result of the last chooseGoal: whether there were any spots before filtering
    bool lastHadSpots = false;

public:

    ExplorePlanner(ObstacleMap& map, uint32_t seed, Clock clock = {}, ExploreConfig cfg = {});

    // log output, default prints "[Explore] ..." to std::cout
    void setLogger(Logger newLogger);
    // forgets everything from a previous run: goal, trail, failed goals, timers, counters
    void reset();
    ExploreCommand step(const ExplorePose& pose);

    State state() const { return currentState; }
    bool hasGoal() const { return hasGoal_; }
    std::pair<float, float> goal() const { return {goalX, goalY}; }
    float goalElapsed() const { return goalElapsed_; }
    const std::deque<std::pair<float, float>>& trail() const { return trail_; }
    const std::deque<std::pair<float, float>>& failedGoals() const { return failedGoals_; }
    const ExploreConfig& config() const { return cfg; }

    // picks the best spot as the new goal; false if there is none (nothing reachable, or
    // all of them are near failed goals)
    bool chooseGoal(float rx, float ry);
    float scoreSpot(const ObstacleMap::Spot& spot, float randomPart) const;
    static float wrapAngle(float a);

private:

    ExploreCommand stepImpl(const ExplorePose& pose);
    void startLookAround(float now);
    void markFailed();
    // printed at most every logInterval seconds
    void log(const std::string& message, float now);
    void say(const std::string& message);
};
}

#endif //FOLLOW_EXPLOREPLANNER_H
