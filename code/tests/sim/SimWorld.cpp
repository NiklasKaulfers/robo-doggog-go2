#include "SimWorld.h"

#include <algorithm>
#include <cmath>
#include "opencv2/imgcodecs.hpp"
#include "tracking/CameraGeometry.h"
#include <fakesdk/bus.hpp>
#include "unitree/idl/go2/HeightMap_.hpp"
#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/idl/go2/WirelessController_.hpp"

using robo::sdk::tracking::CameraGeometry;

namespace sim {

static constexpr auto TICK = std::chrono::milliseconds(20);
// every n-th tick: height map at 10 Hz, remote at 25 Hz
static constexpr int MAP_EVERY = 5, REMOTE_EVERY = 2;
// the LiDAR height map: 6 m x 6 m around the robot in 6 cm cells, with a blind spot under it
static constexpr float MAP_RESOLUTION = 0.06f;
static constexpr int MAP_CELLS = 100;
static constexpr float LIDAR_BLIND_RADIUS = 0.25f;
static constexpr float UNSEEN = 1e9f;
// the camera doesn't detect people further away than this, or closer than this
static constexpr float MAX_VISIBLE_DISTANCE = 8.0f, MIN_VISIBLE_DISTANCE = 0.3f;
// the MoveNet MultiPose output: 6 people of 17 keypoints (y, x, score) and a box
static constexpr int DETECTIONS = 6, VALUES_PER_DETECTION = 17 * 3 + 5;
static constexpr int MODEL_INPUT = 256;

static float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

SimWorld::SimWorld() {
    lastDetections.assign(DETECTIONS * VALUES_PER_DETECTION, 0.0f);
}

SimWorld::~SimWorld() {
    stop();
}

int SimWorld::addWalker(const Walker& walker) {
    std::lock_guard<std::mutex> lock(mutex);
    walkers.push_back(walker);
    closest.push_back(std::hypot(walker.x - body.x, walker.y - body.y));
    return static_cast<int>(walkers.size()) - 1;
}

void SimWorld::setWalker(int index, const Walker& walker) {
    std::lock_guard<std::mutex> lock(mutex);
    walkers.at(index) = walker;
}

Walker SimWorld::walker(int index) const {
    std::lock_guard<std::mutex> lock(mutex);
    return walkers.at(index);
}

void SimWorld::addBox(const Box& box) {
    std::lock_guard<std::mutex> lock(mutex);
    boxes.push_back(box);
}

void SimWorld::setRemote(const Remote& newRemote) {
    std::lock_guard<std::mutex> lock(mutex);
    remote = newRemote;
}

void SimWorld::placeRobot(float x, float y, float yaw) {
    std::lock_guard<std::mutex> lock(mutex);
    body.x = x;
    body.y = y;
    body.yaw = yaw;
    body.z = BODY_HEIGHT + floorHeight(x, y);
}

void SimWorld::setTilt(float roll, float pitch) {
    std::lock_guard<std::mutex> lock(mutex);
    body.roll = roll;
    body.pitch = pitch;
}

Pose SimWorld::pose() const {
    std::lock_guard<std::mutex> lock(mutex);
    return body;
}

void SimWorld::setScript(std::function<void(SimWorld&, double)> newScript) {
    std::lock_guard<std::mutex> lock(mutex);
    script = newScript ? std::make_shared<std::function<void(SimWorld&, double)>>(std::move(newScript)) : nullptr;
}

void SimWorld::start() {
    if (running) return;
    {
        std::lock_guard<std::mutex> lock(mutex);
        body.z = BODY_HEIGHT + floorHeight(body.x, body.y);
    }
    fakesdk::bus().setCamera([this](std::vector<uint8_t>& out) { return camera(out); });
    fakesdk::bus().setDetector([this](std::vector<float>& out) { return detect(out); });
    running = true;
    thread = std::thread([this] { run(); });
}

void SimWorld::stop() {
    if (!running.exchange(false)) return;
    if (thread.joinable()) thread.join();
    fakesdk::bus().setCamera(nullptr);
    fakesdk::bus().setDetector(nullptr);
}

double SimWorld::time() const {
    // one clock for the world and the recorded calls
    return fakesdk::bus().now();
}

bool SimWorld::collided() const {
    std::lock_guard<std::mutex> lock(mutex);
    return !firstCollision.empty();
}

std::string SimWorld::collision() const {
    std::lock_guard<std::mutex> lock(mutex);
    return firstCollision;
}

float SimWorld::closestApproach(int index) const {
    std::lock_guard<std::mutex> lock(mutex);
    return closest.at(index);
}

float SimWorld::distanceTo(int index) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto& w = walkers.at(index);
    return std::hypot(w.x - body.x, w.y - body.y);
}

float SimWorld::cameraDistanceTo(int index) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto& w = walkers.at(index);
    return std::hypot(w.x - (body.x + CAMERA_OFFSET * std::cos(body.yaw)), w.y - (body.y + CAMERA_OFFSET * std::sin(body.yaw)));
}

int SimWorld::framesServed() const {
    std::lock_guard<std::mutex> lock(mutex);
    return frames;
}

int SimWorld::peopleInLastFrame() const {
    std::lock_guard<std::mutex> lock(mutex);
    return lastFramePeople;
}

void SimWorld::run() {
    auto next = std::chrono::steady_clock::now();
    auto last = next;
    int tick = 0;
    // the first state right away, the code under test waits for it
    if (stateStream) publishState();
    if (mapStream) publishMap();
    while (running) {
        next += TICK;
        const auto now = std::chrono::steady_clock::now();
        if (next < now - 5 * TICK) next = now;
        std::this_thread::sleep_until(next);

        const auto current = std::chrono::steady_clock::now();
        const float dt = std::min(std::chrono::duration<float>(current - last).count(), 0.1f);
        last = current;

        std::shared_ptr<std::function<void(SimWorld&, double)>> scripted;
        {
            std::lock_guard<std::mutex> lock(mutex);
            scripted = script;
        }
        if (scripted) (*scripted)(*this, time());
        step(dt);

        if (stateStream) publishState();
        if (tick % MAP_EVERY == 0 && mapStream) publishMap();
        if (tick % REMOTE_EVERY == 0 && remoteStream) publishRemote();
        tick++;
    }
}

void SimWorld::step(float dt) {
    // the velocity the avoidance service accepted last; it listens to the remote instead
    // of the API after UseRemoteCommandFromApi(false), the remote isn't driven here
    const auto avoidance = fakesdk::bus().avoidance();
    std::lock_guard<std::mutex> lock(mutex);
    const float vx = avoidance.apiControl ? avoidance.vx : 0;
    const float vy = avoidance.apiControl ? avoidance.vy : 0;
    const float vyaw = avoidance.apiControl ? avoidance.vyaw : 0;

    body.yaw = wrap(body.yaw + vyaw * dt);
    const float c = std::cos(body.yaw), s = std::sin(body.yaw);
    body.x += (vx * c - vy * s) * dt;
    body.y += (vx * s + vy * c) * dt;
    body.z = BODY_HEIGHT + floorHeight(body.x, body.y);

    for (auto& w : walkers) {
        w.x += w.vx * dt;
        w.y += w.vy * dt;
    }
    checkCollisions();
}

void SimWorld::checkCollisions() {
    const float c = std::cos(body.yaw), s = std::sin(body.yaw);
    for (size_t i = 0; i < walkers.size(); i++)
        closest[i] = std::min(closest[i], std::hypot(walkers[i].x - body.x, walkers[i].y - body.y));
    if (!firstCollision.empty()) return;

    // points spread over the footprint
    for (int i = 0; i <= 6; i++) {
        for (int j = 0; j <= 4; j++) {
            const float along = (i / 6.0f - 0.5f) * BODY_LENGTH, across = (j / 4.0f - 0.5f) * BODY_WIDTH;
            const float px = body.x + along * c - across * s, py = body.y + along * s + across * c;
            for (const auto& b : boxes) {
                if (b.height > STEP_HEIGHT && px >= b.x0 && px <= b.x1 && py >= b.y0 && py <= b.y1) {
                    firstCollision = "box at " + std::to_string(b.x0) + ".." + std::to_string(b.x1) + ", " +
                                     std::to_string(b.y0) + ".." + std::to_string(b.y1) + " (robot at " +
                                     std::to_string(body.x) + ", " + std::to_string(body.y) + ")";
                    return;
                }
            }
            for (size_t k = 0; k < walkers.size(); k++) {
                if (std::hypot(px - walkers[k].x, py - walkers[k].y) < PERSON_RADIUS) {
                    firstCollision = "person " + std::to_string(k) + " (robot at " + std::to_string(body.x) + ", " +
                                     std::to_string(body.y) + ")";
                    return;
                }
            }
        }
    }
}

float SimWorld::floorHeight(float x, float y) const {
    float h = 0;
    for (const auto& b : boxes)
        if (b.height <= STEP_HEIGHT && x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) h = std::max(h, b.height);
    return h;
}

float SimWorld::terrainHeight(float x, float y) const {
    float h = 0;
    for (const auto& b : boxes)
        if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) h = std::max(h, b.height);
    // people's legs show up in the height map too
    for (const auto& w : walkers)
        if (std::hypot(x - w.x, y - w.y) <= PERSON_RADIUS) h = std::max(h, PERSON_HEIGHT);
    return h;
}

bool SimWorld::occluded(float x0, float y0, float x1, float y1) const {
    const float dx = x1 - x0, dy = y1 - y0;
    for (const auto& b : boxes) {
        if (b.height <= SIGHT_HEIGHT) continue;
        // clip the line of sight against the box (Liang-Barsky)
        float t0 = 0, t1 = 1;
        bool inside = true;
        const float p[4] = {-dx, dx, -dy, dy};
        const float q[4] = {x0 - b.x0, b.x1 - x0, y0 - b.y0, b.y1 - y0};
        for (int i = 0; i < 4 && inside; i++) {
            if (p[i] == 0) {
                if (q[i] < 0) inside = false;
                continue;
            }
            const float r = q[i] / p[i];
            if (p[i] < 0) t0 = std::max(t0, r);
            else t1 = std::min(t1, r);
            if (t0 > t1) inside = false;
        }
        if (inside) return true;
    }
    return false;
}

void SimWorld::publishState() {
    unitree_go::msg::dds_::SportModeState_ message;
    {
        std::lock_guard<std::mutex> lock(mutex);
        message.position({body.x, body.y, body.z});
        message.imu_state().rpy({body.roll, body.pitch, body.yaw});
        message.mode(1);
    }
    fakesdk::bus().publish("rt/sportmodestate", message);
}

void SimWorld::publishMap() {
    unitree_go::msg::dds_::HeightMap_ message;
    {
        std::lock_guard<std::mutex> lock(mutex);
        // world aligned and centered on the robot, like the Go2's map
        const float originX = body.x - MAP_CELLS * MAP_RESOLUTION / 2;
        const float originY = body.y - MAP_CELLS * MAP_RESOLUTION / 2;
        message.frame_id("odom");
        message.resolution(MAP_RESOLUTION);
        message.width(MAP_CELLS);
        message.height(MAP_CELLS);
        message.origin({originX, originY});
        std::vector<float> data(static_cast<size_t>(MAP_CELLS) * MAP_CELLS);
        for (int iy = 0; iy < MAP_CELLS; iy++) {
            for (int ix = 0; ix < MAP_CELLS; ix++) {
                const float x = originX + (ix + 0.5f) * MAP_RESOLUTION, y = originY + (iy + 0.5f) * MAP_RESOLUTION;
                const bool blind = std::hypot(x - body.x, y - body.y) < LIDAR_BLIND_RADIUS;
                data[static_cast<size_t>(iy) * MAP_CELLS + ix] = blind ? UNSEEN : terrainHeight(x, y);
            }
        }
        message.data(data);
    }
    fakesdk::bus().publish("rt/utlidar/height_map_array", message);
}

void SimWorld::publishRemote() {
    unitree_go::msg::dds_::WirelessController_ message;
    {
        std::lock_guard<std::mutex> lock(mutex);
        message.lx(remote.lx);
        message.ly(remote.ly);
        message.rx(remote.rx);
        message.ry(remote.ry);
        message.keys(remote.keys);
    }
    fakesdk::bus().publish("rt/wirelesscontroller", message);
}

// the people in a frame as the MoveNet MultiPose model reports them: positions relative to
// the square model input the frame was letterboxed into (see Movenet::processInput)
static void toModelOutput(const std::vector<robo::sdk::ml::PersonDetection>& people, std::vector<float>& out) {
    const float scale = std::min(static_cast<float>(MODEL_INPUT) / SimWorld::FRAME_WIDTH,
                                 static_cast<float>(MODEL_INPUT) / SimWorld::FRAME_HEIGHT);
    const int width = std::min(MODEL_INPUT, static_cast<int>(std::lround(SimWorld::FRAME_WIDTH * scale)));
    const int height = std::min(MODEL_INPUT, static_cast<int>(std::lround(SimWorld::FRAME_HEIGHT * scale)));
    const float padLeft = static_cast<float>((MODEL_INPUT - width) / 2), padTop = static_cast<float>((MODEL_INPUT - height) / 2);
    const auto nx = [&](float x) { return (x * scale + padLeft) / MODEL_INPUT; };
    const auto ny = [&](float y) { return (y * scale + padTop) / MODEL_INPUT; };

    out.assign(DETECTIONS * VALUES_PER_DETECTION, 0.0f);
    for (size_t i = 0; i < people.size() && i < DETECTIONS; i++) {
        const auto& p = people[i];
        float* d = &out[i * VALUES_PER_DETECTION];
        for (int k = 0; k < robo::sdk::ml::PersonDetection::KEYPOINT_COUNT; k++) {
            // MoveNet order per keypoint: y (row), x (column), score
            d[3 * k + 0] = ny(p.keypoints[k].y);
            d[3 * k + 1] = nx(p.keypoints[k].x);
            d[3 * k + 2] = p.confidence[k];
        }
        d[51] = ny(p.box.y);
        d[52] = nx(p.box.x);
        d[53] = ny(p.box.y + p.box.height);
        d[54] = nx(p.box.x + p.box.width);
        d[55] = p.score;
    }
}

int32_t SimWorld::camera(std::vector<uint8_t>& out) {
    const int32_t error = cameraError;
    if (error != 0) return error;
    if (cameraGarbage) {
        out.assign(2048, 0x5a);
        std::lock_guard<std::mutex> lock(mutex);
        lastDetections.assign(DETECTIONS * VALUES_PER_DETECTION, 0.0f);
        return 0;
    }

    cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, testutil::BACKGROUND);
    {
        std::lock_guard<std::mutex> lock(mutex);
        struct Seen {
            float distance;
            testutil::Person drawn;
        };
        std::vector<Seen> seen;
        const float cameraX = body.x + CAMERA_OFFSET * std::cos(body.yaw);
        const float cameraY = body.y + CAMERA_OFFSET * std::sin(body.yaw);
        for (const auto& w : walkers) {
            if (w.hidden) continue;
            const float dx = w.x - cameraX, dy = w.y - cameraY;
            const float distance = std::hypot(dx, dy);
            if (distance < MIN_VISIBLE_DISTANCE || distance > MAX_VISIBLE_DISTANCE) continue;
            const float bearing = wrap(std::atan2(dy, dx) - body.yaw);
            if (std::abs(bearing) > CameraGeometry::FOV / 2) continue;
            if (occluded(cameraX, cameraY, w.x, w.y)) continue;
            // the camera the code assumes: box width inversely proportional to the distance
            const float width = CameraGeometry::CALIBRATION_DISTANCE * CameraGeometry::BOX_WIDTH_AT_CALIBRATION / distance * FRAME_WIDTH;
            const float centerX = (0.5f - bearing / CameraGeometry::FOV) * FRAME_WIDTH;
            seen.push_back({distance, testutil::Person {centerX - width / 2, w.shirt, w.trousers, width / 200}});
        }
        // far people first, nearer ones are drawn over them
        std::sort(seen.begin(), seen.end(), [](const Seen& a, const Seen& b) { return a.distance > b.distance; });
        for (const auto& s : seen) s.drawn.draw(frame);

        // the detector reports the nearest ones; keypoints outside the frame aren't seen
        std::vector<robo::sdk::ml::PersonDetection> detected;
        for (auto it = seen.rbegin(); it != seen.rend() && detected.size() < DETECTIONS; ++it) {
            auto detection = it->drawn.detection();
            for (int k = 0; k < robo::sdk::ml::PersonDetection::KEYPOINT_COUNT; k++) {
                const auto& kp = detection.keypoints[k];
                const bool inFrame = kp.x >= 0 && kp.y >= 0 && kp.x < FRAME_WIDTH && kp.y < FRAME_HEIGHT;
                if (detection.confidence[k] > 0 && !inFrame) detection.confidence[k] = 0.05f;
            }
            detected.push_back(detection);
        }
        toModelOutput(detected, lastDetections);
        frames++;
        lastFramePeople = static_cast<int>(detected.size());
    }
    cv::imencode(".jpg", frame, out, {cv::IMWRITE_JPEG_QUALITY, 90});
    return 0;
}

bool SimWorld::detect(std::vector<float>& output) {
    if (detectorFails) return false;
    std::lock_guard<std::mutex> lock(mutex);
    output = lastDetections;
    return true;
}
}
