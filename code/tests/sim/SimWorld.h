#ifndef FOLLOW_TESTS_SIMWORLD_H
#define FOLLOW_TESTS_SIMWORLD_H

// A small simulated world for the robot code, in place of the Go2 and its sensors.
//
// The robot moves with the velocity the (fake) obstacle avoidance service last accepted;
// people walk; boxes are walls, furniture or stair steps. The world publishes what the
// Go2 would over the fake DDS (fakesdk::Bus): rt/sportmodestate (50 Hz), the LiDAR
// height map rt/utlidar/height_map_array (10 Hz) and the remote rt/wirelesscontroller
// (25 Hz). The front camera renders the people it sees as drawn figures (JPEG, on
// request), and the pose detector returns exactly those people in MoveNet MultiPose
// output layout. Everything runs in real time on its own thread, like the SDK's DDS threads.
//
// The obstacle avoidance service itself is not simulated: the robot goes wherever the
// code under test sends it, so collisions here are the code's own planning failures.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "opencv2/core.hpp"
#include "DrawnPerson.h"

namespace sim {

// an axis-aligned box standing on the floor: a wall, furniture, or a stair step when low
struct Box {
    float x0, y0, x1, y1;
    float height = 1.0f;
};

// a person: feet at (x, y) in the world, walking with (vx, vy) m/s
struct Walker {
    float x = 0, y = 0;
    float vx = 0, vy = 0;
    cv::Scalar shirt = testutil::RED, trousers = testutil::JEANS;
    // the camera doesn't see them (e.g. they stepped behind something), the LiDAR still does
    bool hidden = false;
};

// what the remote controller sends
struct Remote {
    float lx = 0, ly = 0, rx = 0, ry = 0;
    uint16_t keys = 0;
};

// the robot's body as the state messages report it (odometry frame)
struct Pose {
    float x = 0, y = 0, yaw = 0;
    float z = 0;
    float roll = 0, pitch = 0;
};

// remote buttons, see unitree_joystick.hpp
constexpr uint16_t KEY_R1 = 1 << 0, KEY_L1 = 1 << 1, KEY_Y = 1 << 11;

class SimWorld {
public:
    static constexpr int FRAME_WIDTH = 1280, FRAME_HEIGHT = 720;
    // body height above the floor it stands on
    static constexpr float BODY_HEIGHT = 0.32f;
    // the robot's footprint for collisions (m), Go2 standing: 0.70 x 0.31
    static constexpr float BODY_LENGTH = 0.7f, BODY_WIDTH = 0.31f;
    // the front camera sits in the head, this far ahead of the body center (odometry
    // position); the distances the code estimates from it are measured from there
    static constexpr float CAMERA_OFFSET = 0.3f;
    static constexpr float PERSON_RADIUS = 0.2f, PERSON_HEIGHT = 1.7f;
    // boxes up to this high are steps the robot walks onto, higher ones are obstacles
    static constexpr float STEP_HEIGHT = 0.22f;
    // boxes higher than this hide people from the camera
    static constexpr float SIGHT_HEIGHT = 0.5f;

    SimWorld();
    ~SimWorld();
    SimWorld(const SimWorld&) = delete;
    SimWorld& operator=(const SimWorld&) = delete;

    // --- the scene; everything may be changed while running ---
    int addWalker(const Walker& walker);
    void setWalker(int index, const Walker& walker);
    Walker walker(int index) const;
    void addBox(const Box& box);
    void setRemote(const Remote& remote);
    void placeRobot(float x, float y, float yaw);
    // body tilt (rad), e.g. like on stairs; positive pitch = nose down
    void setTilt(float roll, float pitch);
    Pose pose() const;

    // --- sensors that can fail ---
    std::atomic<bool> stateStream {true};
    std::atomic<bool> mapStream {true};
    std::atomic<bool> remoteStream {true};
    // GetImageSample result: 0 = a picture, else this error code
    std::atomic<int32_t> cameraError {0};
    // the camera returns bytes that are no image
    std::atomic<bool> cameraGarbage {false};
    std::atomic<bool> detectorFails {false};

    // called on the world thread every tick with time(), for timed changes to the scene
    void setScript(std::function<void(SimWorld&, double)> script);

    void start();
    // stops the thread and disconnects the camera and detector; the code under test must
    // not use them anymore (stop it first)
    void stop();
    // seconds on the fakesdk::Bus clock (since its last reset), the time of the recorded calls
    double time() const;

    // --- what happened ---
    // the robot's body touched a box higher than a step or a person
    bool collided() const;
    std::string collision() const;
    // closest the robot's center came to a walker, and its distance now
    float closestApproach(int walker) const;
    float distanceTo(int walker) const;
    // distance from the camera, what the code estimates and keeps at its following distance
    float cameraDistanceTo(int walker) const;
    // camera frames served, and how many people were in the last one
    int framesServed() const;
    int peopleInLastFrame() const;

private:
    void run();
    void step(float dt);
    void publishState();
    void publishMap();
    void publishRemote();
    int32_t camera(std::vector<uint8_t>& out);
    bool detect(std::vector<float>& output);
    // these need the mutex
    float floorHeight(float x, float y) const;
    float terrainHeight(float x, float y) const;
    bool occluded(float x0, float y0, float x1, float y1) const;
    void checkCollisions();

    mutable std::mutex mutex;
    std::vector<Walker> walkers;
    std::vector<Box> boxes;
    Remote remote;
    Pose body;
    // shared, so it can run outside the mutex and still keep its state between ticks
    std::shared_ptr<std::function<void(SimWorld&, double)>> script;
    // the people in the last camera frame, MoveNet MultiPose output layout
    std::vector<float> lastDetections;
    int frames = 0;
    int lastFramePeople = 0;
    std::string firstCollision;
    std::vector<float> closest;

    std::thread thread;
    std::atomic<bool> running {false};
};
}

#endif //FOLLOW_TESTS_SIMWORLD_H
