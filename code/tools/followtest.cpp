// Self-test of the FollowPlanner (follow one person, search for them when lost) with a
// simulated robot, simulated people and simulated terrain: people are drawn into camera
// frames where they stand relative to the robot, the robot integrates the returned
// commands into its pose, and the LiDAR height map is made from the terrain around it.
// Covers following, bystanders, losing the target, Pursue, LookAround, Sweep, GaveUp,
// finding them again in every phase, paused time and the operator reset. Faces can't be
// drawn, those rules are covered by trackertest. Runs without the robot:
//
//   cd code && ./build-tools/followtest
//
// (needs ./res with the face models, like the app). Exit code 0 when all checks pass.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgproc.hpp"
#include "../src/core/ObstacleMap.h"
#include "../src/follow/FollowPlanner.h"
#include "../src/ml/PersonDetection.h"
#include "../src/tracking/CameraGeometry.h"

using robo::sdk::core::HeightMapFrame;
using robo::sdk::core::ObstacleMap;
using robo::sdk::follow::FollowPlanner;
using robo::sdk::ml::PersonDetection;
using robo::sdk::tracking::CameraGeometry;
using robo::sdk::tracking::RobotPose;
using Phase = FollowPlanner::Phase;
using namespace std::chrono_literals;
// terrain height at a world point, NAN where the LiDAR sees nothing
using Terrain = std::function<float(float, float)>;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok    " : "  FAIL  ") << what << std::endl;
    if (!condition) failures++;
}

static std::string num(float v) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.2f", v);
    return buffer;
}

static constexpr int FRAME_WIDTH = 1280, FRAME_HEIGHT = 720;
static constexpr float PI = 3.14159265f;
// terrain at least this high blocks the camera's view, lower obstacles only the way
static constexpr float VIEW_BLOCKING_HEIGHT = 1.2f;

static const cv::Scalar RED(40, 40, 220), BLUE(160, 60, 30), GREEN(60, 180, 60), BLACK(30, 30, 30),
    YELLOW(40, 200, 220), PURPLE(180, 40, 180);

static float flat(float, float) { return 0; }

static bool inBox(float x, float y, float x0, float x1, float y0, float y1) {
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

// a person standing or walking in the world, seen from behind
struct Walker {
    float x = 0, y = 0;
    cv::Scalar shirt, trousers;
    bool present = true;
};

// one person in the image, drawn like in trackertest: at scale 1 the body is 200 x 600 px
struct Drawn {
    float left, scale;
    cv::Scalar shirt, trousers;

    float top() const { return 700 - 600 * scale; }

    PersonDetection detection() const {
        PersonDetection p;
        p.box = cv::Rect2f(left, top(), 200 * scale, 600 * scale) & cv::Rect2f(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        p.score = 0.8f;
        auto set = [&](int k, float px, float py) {
            p.keypoints[k] = cv::Point2f(left + px * scale, top() + (py - 100) * scale);
            p.confidence[k] = 0.9f;
        };
        set(PersonDetection::LEFT_SHOULDER, 40, 220);
        set(PersonDetection::RIGHT_SHOULDER, 160, 220);
        set(PersonDetection::LEFT_HIP, 60, 420);
        set(PersonDetection::RIGHT_HIP, 140, 420);
        set(PersonDetection::LEFT_KNEE, 65, 560);
        set(PersonDetection::RIGHT_KNEE, 135, 560);
        set(PersonDetection::LEFT_ANKLE, 65, 690);
        set(PersonDetection::RIGHT_ANKLE, 135, 690);
        return p;
    }

    void draw(cv::Mat& frame) const {
        auto rect = [&](float px, float py, float w, float h) {
            return cv::Rect(cv::Point(left + px * scale, top() + (py - 100) * scale), cv::Size(w * scale, h * scale));
        };
        cv::rectangle(frame, rect(30, 200, 140, 230), shirt, cv::FILLED);
        cv::rectangle(frame, rect(40, 430, 120, 270), trousers, cv::FILLED);
    }
};

// the simulated robot, its surroundings and the planner under test
struct World {
    ObstacleMap map;
    float time = 0;
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    Terrain terrain;
    FollowPlanner planner;
    RobotPose pose;
    std::vector<Walker> people;
    std::vector<std::string> logs;
    FollowPlanner::Step last;
    // where each person was in the last frame, empty box if not in view
    std::vector<cv::Rect2f> lastBoxes;
    // the robot turns this much slower than commanded (e.g. slippery floor)
    float turnFactor = 1;
    // largest commands seen, for the speed caps
    float maxForward = 0, maxBackward = 0, maxYaw = 0;

    explicit World(Terrain newTerrain = flat, uint32_t seed = 1) : terrain(std::move(newTerrain)), planner(map, seed) {
        map.setClock([this] { return time; });
        planner.setLogger([this](const std::string& message, bool) { logs.push_back(message); });
        if (!planner.init("./res")) {
            std::cerr << "run from the code folder, ./res with the face models is needed" << std::endl;
            std::exit(1);
        }
    }

    // one LiDAR frame of the terrain around the robot
    void observe() {
        const float half = 5.0f, res = 0.1f;
        const int n = static_cast<int>(std::lround(2 * half / res));
        std::vector<float> data(static_cast<size_t>(n) * n);
        HeightMapFrame frame;
        frame.resolution = res;
        frame.width = frame.height = n;
        frame.originX = pose.x - n * res / 2;
        frame.originY = pose.y - n * res / 2;
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                const float h = terrain(frame.originX + (x + 0.5f) * res, frame.originY + (y + 0.5f) * res);
                data[y * n + x] = std::isnan(h) ? 1e9f : h;
            }
        }
        frame.data = data.data();
        frame.size = data.size();
        map.integrate(frame);
    }

    bool lineOfSight(float x, float y) const {
        const float d = std::hypot(x - pose.x, y - pose.y);
        for (float s = 0.2f; s < d - 0.2f; s += 0.05f) {
            const float h = terrain(pose.x + (x - pose.x) * s / d, pose.y + (y - pose.y) * s / d);
            if (h >= VIEW_BLOCKING_HEIGHT) return false;
        }
        return true;
    }

    // where a person shows up in the camera, false if not in view
    bool seen(const Walker& w, Drawn& out) const {
        if (!w.present) return false;
        const float d = std::hypot(w.x - pose.x, w.y - pose.y);
        const float angle = CameraGeometry::wrapAngle(std::atan2(w.y - pose.y, w.x - pose.x) - pose.yaw);
        if (d < 0.3f || d > 8 || std::abs(angle) > CameraGeometry::FOV / 2 - 0.05f) return false;
        if (!lineOfSight(w.x, w.y)) return false;
        const float width = CameraGeometry::CALIBRATION_DISTANCE * CameraGeometry::BOX_WIDTH_AT_CALIBRATION / d * FRAME_WIDTH;
        const float center = (0.5f - angle / CameraGeometry::FOV) * FRAME_WIDTH;
        out = {center - width / 2, width / 200, w.shirt, w.trousers};
        return true;
    }

    bool sees(size_t person) const {
        Drawn d {};
        return seen(people[person], d);
    }

    // advances everything by dt and runs the planner once
    const FollowPlanner::Step& tick(float dt = 0.1f, bool move = true) {
        time += dt;
        now += std::chrono::microseconds(static_cast<int64_t>(dt * 1e6f));
        observe();

        cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cv::Scalar(120, 120, 120));
        std::vector<std::pair<float, Drawn>> inView;
        lastBoxes.assign(people.size(), cv::Rect2f());
        for (size_t i = 0; i < people.size(); i++) {
            Drawn d {};
            if (!seen(people[i], d)) continue;
            inView.emplace_back(std::hypot(people[i].x - pose.x, people[i].y - pose.y), d);
            lastBoxes[i] = d.detection().box;
        }
        // far people first, near ones are drawn over them
        std::sort(inView.begin(), inView.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<PersonDetection> detections;
        for (const auto& [distance, d] : inView) {
            d.draw(frame);
            const PersonDetection p = d.detection();
            if (p.box.width > 0 && p.box.height > 0) detections.push_back(p);
        }

        last = planner.step(detections, frame, pose, now);
        maxForward = std::max(maxForward, last.x);
        maxBackward = std::max(maxBackward, -last.x);
        maxYaw = std::max(maxYaw, std::abs(last.z));
        if (move) {
            pose.yaw = CameraGeometry::wrapAngle(pose.yaw + last.z * turnFactor * dt);
            pose.x += last.x * std::cos(pose.yaw) * dt;
            pose.y += last.x * std::sin(pose.yaw) * dt;
        }
        return last;
    }

    void run(float seconds, const std::function<void()>& each = {}) {
        const int n = static_cast<int>(std::lround(seconds / 0.1f));
        for (int i = 0; i < n; i++) {
            if (each) each();
            tick();
        }
    }

    float distanceTo(size_t person) const { return std::hypot(people[person].x - pose.x, people[person].y - pose.y); }
    // whether the planner follows this person: it is following and they were the target
    // in the last frame
    bool following(size_t person) const {
        if (last.phase != Phase::Following || !last.target.visible || person >= lastBoxes.size()) return false;
        const cv::Rect2f& box = lastBoxes[person];
        return !box.empty() && std::abs(last.target.target.box.x - box.x) < 2;
    }

    bool logged(const std::string& part) const {
        return std::any_of(logs.begin(), logs.end(), [&](const auto& l) { return l.find(part) != std::string::npos; });
    }
};

static void testWaiting() {
    std::cout << "no target yet" << std::endl;
    World w;
    bool still = true;
    for (int i = 0; i < 20; i++) {
        const auto& s = w.tick();
        still = still && s.x == 0 && s.z == 0;
    }
    check(w.last.phase == Phase::WaitingForTarget, "phase is WaitingForTarget");
    check(still, "stands still while nobody is there");
}

static void testFollowing() {
    std::cout << "following the first person" << std::endl;
    World w;
    w.people.push_back({2.5f, 0.3f, RED, BLUE});
    w.run(1);
    check(w.following(0), "locks on to the first person who stays in view");
    check(w.last.x > 0, "walks toward them");
    w.run(15);
    const float d = w.distanceTo(0);
    check(d > 0.4f && d < 0.7f, "keeps the following distance (" + num(d) + " m)");
    check(w.following(0), "still following them");

    // they step back to 2 m, then walk to the robot's left at 0.4 m/s
    const float yaw0 = w.pose.yaw;
    int left = 0, right = 0;
    for (int i = 0; i < 15; i++) {
        w.people[0].x += 0.1f * std::cos(yaw0);
        w.people[0].y += 0.1f * std::sin(yaw0);
        w.tick(0.1f, false);
    }
    // facing them again
    w.run(1);
    for (int i = 0; i < 20; i++) {
        w.people[0].x -= 0.04f * std::sin(yaw0);
        w.people[0].y += 0.04f * std::cos(yaw0);
        w.tick();
        left += w.last.z > 0;
        right += w.last.z < 0;
    }
    check(left > 15 && right == 0, "turns toward a person walking to the left (" + std::to_string(left) + " left, " + std::to_string(right) + " right)");
    check(w.following(0), "and keeps them as the target");
    check(w.maxForward <= 0.6f && w.maxBackward <= 0.6f && w.maxYaw <= 0.6f, "speed caps kept (0.6 m/s, 0.6 rad/s)");
}

static void testObstacle() {
    std::cout << "an obstacle between robot and person" << std::endl;
    // a low box (the camera sees over it, the robot can't walk over it) right in the way
    const auto box = [](float x, float y) { return inBox(x, y, 1.2f, 1.5f, -0.7f, 0.7f) ? 0.5f : 0.0f; };
    World w(box);
    w.people.push_back({3.0f, 0, RED, BLUE});
    bool collided = false;
    float maxSide = 0;
    w.run(20, [&] {
        collided = collided || inBox(w.pose.x, w.pose.y, 1.2f - 0.15f, 1.5f + 0.15f, -0.7f - 0.15f, 0.7f + 0.15f);
        maxSide = std::max(maxSide, std::abs(w.pose.y));
    });
    check(!collided, "never walks into the obstacle");
    check(maxSide > 0.7f, "heads around it (" + num(maxSide) + " m to the side)");
}

static void testBystander() {
    std::cout << "a bystander in other clothes is in view all the time" << std::endl;
    World w;
    w.people.push_back({2.5f, 0.3f, RED, BLUE});
    w.run(2);
    check(w.following(0), "following the first person");
    // closer than the target, and stays
    w.people.push_back({1.5f, -0.5f, GREEN, BLACK});
    bool bystanderFollowed = false;
    w.run(15, [&] {
        // the target slowly walks off to the left and back
        w.people[0].y = 0.3f + 1.0f * std::sin(w.time * 0.3f);
        bystanderFollowed = bystanderFollowed || w.following(1);
    });
    check(!bystanderFollowed, "the bystander is never followed");
    check(w.following(0), "still following the target");
}

static void testStrangerAfterEarlyLoss() {
    std::cout << "the target is lost before anything was learned, then a stranger shows up" << std::endl;
    World w;
    w.people.push_back({2.5f, 0, RED, BLUE});
    w.run(0.3f);
    check(w.following(0), "the first person is the target");
    w.people[0].present = false;
    w.run(4);
    w.people.push_back({2.0f, 0.2f, GREEN, BLACK});
    bool strangerFollowed = false;
    w.run(20, [&] { strangerFollowed = strangerFollowed || w.following(1) || w.last.phase == Phase::Following; });
    check(!strangerFollowed, "the stranger is not followed");
}

// follows the target for a few seconds while they walk to the robot's left (+y, side 1)
// or right (-y, side -1) at 0.5 m/s, then they vanish (e.g. through a door). Returns
// where they were last seen
static std::pair<float, float> loseWalkingTarget(World& w, int side = 1) {
    w.people.push_back({2.0f, -1.0f * side, RED, BLUE});
    w.run(2);
    w.run(3, [&] { w.people[0].y += 0.05f * side; });
    const std::pair<float, float> lastSeen {w.people[0].x, w.people[0].y};
    w.people[0].present = false;
    return lastSeen;
}

// runs until the phase is not `phase` anymore (at most `limit` seconds); returns the
// yaw turned meanwhile (signed, from the robot's pose) and whether every yaw command
// had the sign `sign` (0 = don't care) without forward motion
static float runPhase(World& w, Phase phase, float limit, int sign, bool& onlyTurning) {
    float turned = 0;
    onlyTurning = true;
    for (int i = 0; i < limit / 0.1f && w.last.phase == phase; i++) {
        const float yaw = w.pose.yaw;
        w.tick();
        if (w.last.phase != phase) break;
        turned += CameraGeometry::wrapAngle(w.pose.yaw - yaw);
        if (w.last.x != 0 || (sign != 0 && w.last.z * sign <= 0)) onlyTurning = false;
    }
    return turned;
}

static void testPursueAndLookAround(float turnFactor, int side) {
    std::cout << "the target walks out of view to the " << (side > 0 ? "left" : "right")
              << (turnFactor < 1 ? " (the robot turns slower than commanded)" : "") << std::endl;
    World w;
    w.turnFactor = turnFactor;
    const auto lastSeen = loseWalkingTarget(w, side);
    check(w.last.phase == Phase::Following, "following until they vanish");
    w.run(0.5f);
    check(w.last.phase == Phase::Pursue, "lost within half a second: Pursue");
    const auto startPose = w.pose;
    while (w.last.phase == Phase::Pursue && w.time < 60) w.tick();
    check(w.last.phase == Phase::LookAround, "arrives and looks around");
    // they were heading along y at 0.5 m/s: expected about 1 m (2 s of walking) further
    const float further = (w.pose.y - lastSeen.second) * side;
    const float fromPredicted = std::hypot(w.pose.x - lastSeen.first, w.pose.y - (lastSeen.second + 1.0f * side));
    // (the expected spot is approximate: the tracker estimates their speed from the box)
    check(fromPredicted < 1.2f && further > 0.3f,
          "Pursue went past where they vanished, the way they walked (" + num(fromPredicted) + " m from the expected spot, " +
          num(further) + " m further)");
    check(std::hypot(w.pose.x - startPose.x, w.pose.y - startPose.y) > 0.5f, "and actually walked there");
    bool onlyTurning = false;
    const float turned = runPhase(w, Phase::LookAround, 60, side, onlyTurning) * side;
    check(onlyTurning, std::string("LookAround turns on the spot toward the side they left (") + (side > 0 ? "left" : "right") + ")");
    check(turned > 2 * PI - 0.15f && turned < 2 * PI + 0.3f, "one full turn (" + num(turned) + " rad)");
    check(w.last.phase == Phase::Sweep, "then sweeps the area");
    check(w.logged("LookAround") && w.logged("Pursue"), "phase changes are logged");
}

static void testPursueBlocked() {
    std::cout << "the spot they were heading to is walled off" << std::endl;
    // a closed pen (low walls) around where they were heading, 1 m past where they vanish
    const auto pen = [](float x, float y) {
        const bool inside = inBox(x, y, 1.4f, 2.6f, 1.0f, 2.4f), core = inBox(x, y, 1.6f, 2.4f, 1.2f, 2.2f);
        return inside && !core ? 0.5f : 0.0f;
    };
    World w(pen);
    loseWalkingTarget(w);
    w.run(0.5f);
    check(w.last.phase == Phase::Pursue, "Pursue");
    float pursued = 0;
    bool collided = false;
    while (w.last.phase == Phase::Pursue && pursued < 30) {
        w.tick();
        pursued += 0.1f;
        collided = collided || pen(w.pose.x, w.pose.y) > 0;
    }
    check(w.last.phase == Phase::LookAround, "gives up on the spot and looks around");
    check(pursued > 7 && pursued < 9.5f, "after the pursue timeout (" + num(pursued) + " s)");
    check(!collided, "without walking into the walls");
    check(w.logged("could not get where they were heading"), "the timeout is logged");
}

static void testSweepArea(uint32_t seed, int& onWalkingSide, int& samples, bool& inside, float& farthest) {
    World w(flat, seed);
    const auto lastSeen = loseWalkingTarget(w);
    while (w.last.phase != Phase::Sweep && w.time < 120) w.tick();
    const float start = w.last.searchSeconds;
    for (int i = 0; i < 600 && w.last.phase == Phase::Sweep; i++) {
        w.tick();
        const float d = std::hypot(w.pose.x - lastSeen.first, w.pose.y - lastSeen.second);
        // the area starts at 2 m and grows by 0.1 m/s; the robot's own position may lag
        // its goal a little
        const float radius = std::min(2.0f + 0.1f * (w.last.searchSeconds - start), 8.0f);
        inside = inside && d <= radius + 0.8f;
        farthest = std::max(farthest, d);
        samples++;
        // they walked toward +y
        onWalkingSide += w.pose.y > lastSeen.second;
    }
}

static void testSweep() {
    std::cout << "sweeping around where they were last seen" << std::endl;
    int onWalkingSide = 0, samples = 0;
    bool inside = true;
    float farthest = 0;
    for (uint32_t seed = 1; seed <= 3; seed++) testSweepArea(seed, onWalkingSide, samples, inside, farthest);
    check(samples > 1000, "swept for a minute per run");
    check(inside, "the robot stays inside the growing search area");
    check(farthest > 3.5f, "the area grows over time (got " + num(farthest) + " m away)");
    const float share = static_cast<float>(onWalkingSide) / std::max(samples, 1);
    check(share > 0.65f, "mostly on the side they walked to (" + num(share * 100) + " % of the time)");
}

// lets the target show up 2 m in front of the robot, returns after how many frames the
// planner follows them again (99 = not within 2 s)
static int reappear(World& w) {
    w.people[0] = {w.pose.x + 2 * std::cos(w.pose.yaw), w.pose.y + 2 * std::sin(w.pose.yaw), RED, BLUE};
    for (int i = 1; i <= 20; i++) {
        w.tick();
        if (w.following(0)) return i;
    }
    return 99;
}

static void testFoundAgain() {
    std::cout << "the target shows up again" << std::endl;
    for (const Phase phase : {Phase::Pursue, Phase::LookAround, Phase::Sweep}) {
        World w;
        loseWalkingTarget(w);
        while (w.last.phase != phase && w.time < 120) w.tick();
        w.run(1);
        const bool inPhase = w.last.phase == phase;
        const int frames = reappear(w);
        check(inPhase && frames <= 2, std::string("during ") + FollowPlanner::phaseName(phase) + ": following again after " +
                                          std::to_string(frames) + " frame(s)");
    }
}

static void testGaveUp() {
    std::cout << "the search time runs out" << std::endl;
    World w;
    loseWalkingTarget(w);
    bool stoppedInTime = true;
    while (w.last.phase != Phase::GaveUp && w.time < 200) {
        w.tick();
        if (w.last.searchSeconds > 121) stoppedInTime = false;
    }
    check(w.last.phase == Phase::GaveUp && stoppedInTime, "GaveUp after the 2 minute search limit (" + num(w.last.searchSeconds) + " s)");
    check(w.logged("GaveUp"), "logged");
    bool still = true;
    w.run(10, [&] { still = still && w.last.x == 0 && w.last.z == 0; });
    check(still && w.last.phase == Phase::GaveUp, "stands still afterwards");
    const int frames = reappear(w);
    check(frames <= 2, "follows them again when they show up (" + std::to_string(frames) + " frame(s))");
    check(w.logged("found the target again"), "finding them again is logged");
}

static void testPaused() {
    std::cout << "paused time" << std::endl;
    World w;
    loseWalkingTarget(w);
    while (w.last.phase != Phase::Sweep && w.time < 120) w.tick();
    w.run(5);
    const float before = w.last.searchSeconds;
    // the remote controller took over for ten minutes
    w.planner.pause();
    w.tick(600, false);
    check(std::abs(w.last.searchSeconds - before) < 1e-3f, "a pause doesn't count toward the search (" + num(before) + " -> " +
                                                              num(w.last.searchSeconds) + " s)");
    check(w.last.phase == Phase::Sweep, "still sweeping, not given up");
    // a long gap without a pause (e.g. a hanging camera) counts as one short step
    const float next = w.last.searchSeconds;
    w.tick(60, false);
    check(w.last.searchSeconds - next <= 0.5f + 1e-3f, "a 60 s gap counts at most 0.5 s");
}

static void testReset() {
    std::cout << "operator reset" << std::endl;
    World w;
    w.people.push_back({2.5f, 0, RED, BLUE});
    w.run(3);
    check(w.following(0), "following the first person");
    w.people[0].present = false;
    w.run(1);
    check(w.last.phase == Phase::Pursue, "searching for them");
    w.planner.reset();
    w.tick();
    check(w.last.phase == Phase::WaitingForTarget && w.last.x == 0 && w.last.z == 0, "after the reset: waiting, standing still");
    w.people.push_back({w.pose.x + 2 * std::cos(w.pose.yaw), w.pose.y + 2 * std::sin(w.pose.yaw), GREEN, BLACK});
    w.run(1);
    check(w.following(1), "the next person becomes the target");
}

int main() {
    testWaiting();
    testFollowing();
    testObstacle();
    testBystander();
    testStrangerAfterEarlyLoss();
    testPursueAndLookAround(1.0f, 1);
    testPursueAndLookAround(0.5f, 1);
    testPursueAndLookAround(1.0f, -1);
    testPursueBlocked();
    testSweep();
    testFoundAgain();
    testGaveUp();
    testPaused();
    testReset();

    std::cout << (failures == 0 ? "all checks passed" : std::to_string(failures) + " check(s) failed") << std::endl;
    return failures == 0 ? 0 : 1;
}
