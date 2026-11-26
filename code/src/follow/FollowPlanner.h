#ifndef FOLLOW_FOLLOWPLANNER_H
#define FOLLOW_FOLLOWPLANNER_H

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "../core/ExplorePlanner.h"
#include "../core/ObstacleMap.h"
#include "../ml/PersonDetection.h"
#include "../tracking/TargetTracker.h"

namespace robo::sdk::follow {

// Decides how the robot follows one person and searches for them when they are lost.
// Needs no robot (camera frames, detections, pose and time are passed in), so it can be
// tested with a simulated one. Not thread safe, call from one thread.
//
// Phases:  Following -> lost -> Pursue -> LookAround -> Sweep -> GaveUp
//  - Following: chase the target at the following distance, around obstacles
//  - Pursue: walk to where the target was heading (last position plus walking direction)
//  - LookAround: one full turn on the spot, measured by the robot's actual rotation,
//    starting toward the side the target left
//  - Sweep: roam inside a search area around where they were last seen, leaning toward
//    the direction they walked; the area grows over time
//  - GaveUp: after the total search time, stand still and keep watching
// Seeing the target again in any phase returns to Following. Who the target is, is the
// TargetTracker's decision.
class FollowPlanner {

public:

    using Clock = std::chrono::steady_clock;
    // warning = something safety related (e.g. a stale height map), default: std::cerr
    using Logger = std::function<void(const std::string& message, bool warning)>;

    enum class Phase { WaitingForTarget, Following, Pursue, LookAround, Sweep, GaveUp };

    struct Step {
        // forward speed (m/s) and yaw speed (rad/s, positive = left)
        float x = 0, z = 0;
        Phase phase = Phase::WaitingForTarget;
        // what the tracker knows about the target in this frame
        tracking::TargetTracker::Status target;
        // seconds searched since the target was lost (Pursue + LookAround + Sweep), not
        // counting paused time; 0 while following
        float searchSeconds = 0;
    };

    FollowPlanner(core::ObstacleMap& map, uint32_t seed);

    // loads the face models from resourceDir
    bool init(const std::string& resourceDir = "./res");
    // only the person in the photos is followed; false if no face was found in any
    bool loadReferencePhotos(const std::string& folder);
    // log output, default prints "[Follow] ..." to std::cout, warnings to std::cerr
    void setLogger(Logger newLogger);

    // one camera frame: everyone the pose detector found in it, the frame (BGR), the
    // robot pose in the odometry frame and the time it was taken
    const Step& step(const std::vector<ml::PersonDetection>& detections, const cv::Mat& frame,
                     const tracking::RobotPose& pose, Clock::time_point now);

    // the robot was not driven by the planner since the last step (remote controller,
    // exploring, sensor trouble): the time and turning until the next step don't count
    // toward the search
    void pause();
    // operator reset: forget the target and start over, the next person (or the one in
    // the photos) becomes the target
    void reset();

    Phase phase() const { return current.phase; }
    static const char* phaseName(Phase phase);
    // draws the tracker's view and the phase into image (the frame) for debugging
    void drawDebug(cv::Mat& image) const;

private:

    core::ObstacleMap& map;
    tracking::TargetTracker tracker;
    Logger logger;
    Step current;

    bool haveStep = false;
    bool paused = false;
    Clock::time_point lastTime;
    float lastYaw = 0;
    // seconds the planner was in control (pauses and long gaps between steps don't
    // count), the clock of the search timers and of the sweep's ExplorePlanner
    float activeSeconds = 0;
    // seconds in the current search phase
    float phaseSeconds = 0;
    // LookAround: 1 = left, -1 = right, and how far the robot turned that way so far
    float lookDirection = 1;
    float lookTurned = 0;
    // Sweep: where the target was last seen, the direction they walked, and the radius
    // of the search area last logged
    core::ExplorePlanner::SearchFocus sweepArea;
    float sweepLoggedRadius = 0;
    core::ExplorePlanner sweep;
    // 1 = left, -1 = right; detours keep going around the same side
    int lastDetourSide = 1;
    bool mapStale = false;

    // chases a point at distance / angle relative to the robot: keeps the following
    // distance, detours around obstacles, only turns on a stale map
    void chase(float distance, float angle, const tracking::RobotPose& pose);
    // heading (relative to the robot) toward the person that avoids obstacles in the
    // remembered map; clearance is set to how far the robot can walk along it
    float planHeading(float personAngle, float distance, const tracking::RobotPose& pose, float& clearance);
    // the target is lost: decides the search phase and its command
    void search(const tracking::RobotPose& pose, float dt, float turned);
    void startLookAround(const std::string& reason);
    void startSweep(const tracking::RobotPose& pose);
    void enter(Phase phase, const std::string& reason);
    void say(const std::string& message, bool warning = false);
};
}

#endif //FOLLOW_FOLLOWPLANNER_H
