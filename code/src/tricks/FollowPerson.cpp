#include "FollowPerson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include "../ml/movenet/Multipose.h"

using namespace robo::sdk::examples;
using namespace robo::sdk::ml;

// The width of a standing person in normalized image coordinates (0..1) when they
// stand approximately 0.5m away from the Go2's front camera. Movenet's boxes scale
// roughly inversely with the distance to the camera, so comparing the detected box
// width against this value gives an estimate of the current distance between the
// robot and the person. It is calibrated for a ~0.4m wide person (shoulders) in front
// of the Go2's 120° FOV camera and can be tuned per installation.
static constexpr float BOX_WIDTH_AT_0_5M = 0.35f;
static constexpr float TARGET_DISTANCE = 0.5f;
static constexpr float PERSON_CENTER_TOLERANCE = 0.15f;

// stairs are detected by the body tilting forward/backward. Entering and leaving use
// different thresholds (hysteresis) so the robot doesn't toggle on every step edge.
static constexpr float STAIR_PITCH_ENTER = 0.14f;   // ~8 degrees
static constexpr float STAIR_PITCH_EXIT = 0.07f;    // ~4 degrees
static constexpr int STAIR_ENTER_SAMPLES = 3;       // 0.3s tilted -> on stairs
static constexpr int STAIR_EXIT_SAMPLES = 10;       // 1.0s level -> off stairs
// the body also rises/drops by roughly one step height per step
static constexpr float STAIR_HEIGHT_CHANGE = 0.12f;
// on stairs: slow, mostly straight and never backwards
static constexpr float STAIR_MAX_SPEED_UP = 0.35f;
static constexpr float STAIR_MAX_SPEED_DOWN = 0.25f;
static constexpr float STAIR_MAX_YAW = 0.2f;
// tilted sideways this far means the robot is about to slip off a step
static constexpr float STAIR_MAX_ROLL = 0.35f;      // ~20 degrees

// horizontal field of view of the front camera, maps image position to an angle
static constexpr float CAMERA_FOV = 2.09f;          // 120 degrees
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

static constexpr float MIN_CONFIDENCE = 0.15f;
// lost person: their last position and walking velocity predict where they went
static constexpr float PREDICT_HORIZON = 2.0f;      // s, don't extrapolate further than this
static constexpr float MAX_PERSON_SPEED = 1.5f;     // m/s, caps noisy velocity estimates
static constexpr float VELOCITY_SMOOTHING = 0.3f;   // weight of a new velocity measurement
// a detection this close (rad) to the predicted direction is taken as the lost person
static constexpr float REACQUIRE_ANGLE = 0.4f;
// once at the predicted spot, look around toward the side the person left
static constexpr float SEARCH_ARRIVED = 0.6f;
static constexpr float SEARCH_TURN_SPEED = 0.4f;
// after this the robot roams around semi-randomly until the person shows up again
static constexpr float SEARCH_TIMEOUT = 10.0f;
static constexpr int LOST_FRAMES = 3;
// following a specific person: their face confirms them, after that their clothes (as
// learned from the camera) recognize them from behind. How much the clothes must look
// alike to find them again, and how little while following them before it is taken as
// someone else stepping in front. Tune with the logged values.
static constexpr float CLOTHING_SIMILARITY = 0.7f;
static constexpr float TRACK_MIN_SIMILARITY = 0.5f;
// faces are checked every frame while looking for the person, while following them
// only every few frames as the face detection is expensive
static constexpr int FACE_CHECK_INTERVAL = 3;

static float wrapAngle(float a) {
    return std::atan2(std::sin(a), std::cos(a));
}

static float boxCenterX(const movenet::Box& box) {
    return box.xmax - (box.xmax - box.xmin) / 2;
}

// estimate the distance from the robot to the person: the apparent width of a
// standing person scales roughly inversely with their distance to the camera,
// which gives a much more sensitive measurement than the box height here
static float boxDistance(const movenet::Box& box) {
    return TARGET_DISTANCE * BOX_WIDTH_AT_0_5M / (box.xmax - box.xmin);
}

// direction of the person relative to the robot's heading, positive = left
static float boxAngle(const movenet::Box& box) {
    return -(boxCenterX(box) - 0.5f) * CAMERA_FOV;
}

FollowPerson::FollowPerson(core::Control& control, std::string targetImages)
    : roboControl(control), targetImages(std::move(targetImages)) {
    videoClient.Init();
    obstacleMap.init();
}

void FollowPerson::run() {
    runWithMultipose();
}

void FollowPerson::runWithMultipose() {
    movenet::Multipose detector;
    if (!detector.init()) return;

    if (!targetImages.empty()) {
        if (!identifier.load(targetImages)) {
            std::cerr << "[FollowPerson] no face found in the images in " << targetImages << std::endl;
            return;
        }
        identify = true;
        std::cout << "[FollowPerson] waiting to see the face of the person from the images" << std::endl;
    }

    // the first person we saw, which is tracked until it is lost
    bool targetFound = false;
    movenet::Box targetBox {0, 0, 0, 0, 0};
    int countNotFound = 0;
    int frameCount = 0;

    int stairSamples = 0;
    float levelHeight = roboControl.positionZ();

    while (!core::Control::stopRequested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        updateStairMode(stairSamples, levelHeight);

        if (roboControl.exploreComboPressed()) {
            exploring = !exploring;
            if (exploring) {
                explorer.reset();
                std::cout << "[FollowPerson] exploring" << std::endl;
            } else {
                std::cout << "[FollowPerson] stopped exploring, following again" << std::endl;
            }
            // afterwards follow whoever (or the identified person) shows up first
            targetFound = false;
            roamSearching = false;
        }

        // the remote controller always wins: don't send anything while it is used
        if (roboControl.controllerActive()) {
            roboControl.releaseControl();
            moving = false;
            continue;
        }

        // roams until the explore combo is pressed again
        if (exploring) {
            float x, z;
            explorer.step(x, z);
            drive(x, z);
            continue;
        }

        videoClient.GetImageSample(imgData);
        if (!detector.run(imgData)) {
            std::cerr << "[FollowPerson] error in detector.run()" << std::endl;
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        const float rx = roboControl.positionX();
        const float ry = roboControl.positionY();
        const float yaw = roboControl.yaw();

        // where the person should be by now, from their last position and velocity
        const float sinceSeen = std::chrono::duration<float>(now - lastSeen).count();
        const float t = std::min(sinceSeen, PREDICT_HORIZON);
        const float predX = personX + personVX * t;
        const float predY = personY + personVY * t;
        const float predAngle = wrapAngle(std::atan2(predY - ry, predX - rx) - yaw);
        const float predDistance = std::hypot(predX - rx, predY - ry);

        const bool tracking = targetFound && countNotFound == 0 && !roamSearching;
        if (identify) {
            if (!tracking || ++frameCount % FACE_CHECK_INTERVAL == 0) identifier.detectFaces(detector);
            else identifier.clearFaces();
        }

        // find our person among the detections:
        //  - nobody followed yet: the detection with the highest confidence
        //  - still in view: the one whose center stays close to the last box
        //  - out of view: the one closest to the predicted direction
        // with reference images:
        //  - nobody followed yet: only someone whose face matches
        //  - a matching face always wins, wherever it is; a face that clearly doesn't
        //    match is never taken
        //  - out of view: found again by face, or from behind by the clothes learned
        //    while their face was visible
        int match = -1;
        float bestScore = 0;
        PersonIdentifier::Face matchFace = PersonIdentifier::Face::Unknown;
        float matchSimilarity = -1;
        for (int d = 0; d < detector.modelDetectionCount(); d++) {
            const auto box = detector.getBox(d);
            if (box.confidence < MIN_CONFIDENCE) continue;

            float faceSimilarity = -1;
            const auto face = identify ? identifier.face(detector, d, &faceSimilarity) : PersonIdentifier::Face::Unknown;
            const float clothes = identify ? identifier.clothing(detector, d) : -1;
            if (face == PersonIdentifier::Face::Mismatch) continue;

            float score;
            float similarity = -1;
            if (face == PersonIdentifier::Face::Match) {
                score = 2 + faceSimilarity;
                similarity = faceSimilarity;
            } else if (identify && !tracking) {
                if (!targetFound || clothes < CLOTHING_SIMILARITY) {
                    // shortly lost and too little visible to tell: fall back to the direction
                    if (!targetFound || roamSearching || clothes >= 0) continue;
                    const float diff = std::abs(wrapAngle(boxAngle(box) - predAngle));
                    if (diff >= REACQUIRE_ANGLE) continue;
                    score = REACQUIRE_ANGLE - diff;
                } else {
                    score = 1 + clothes;
                    similarity = clothes;
                }
            } else if (!targetFound || roamSearching) {
                // while roaming the prediction is outdated, whoever shows up is followed
                score = box.confidence;
            } else if (countNotFound == 0) {
                // too little visible to tell (-1) is fine, clearly different clothes are not
                if (identify && clothes >= 0 && clothes < TRACK_MIN_SIMILARITY) continue;
                const float dx = boxCenterX(box) - boxCenterX(targetBox);
                const float dy = (box.ymax - (box.ymax - box.ymin) / 2) - (targetBox.ymax - (targetBox.ymax - targetBox.ymin) / 2);
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist >= PERSON_CENTER_TOLERANCE) continue;
                score = PERSON_CENTER_TOLERANCE - dist;
            } else {
                const float diff = std::abs(wrapAngle(boxAngle(box) - predAngle));
                if (diff >= REACQUIRE_ANGLE) continue;
                score = REACQUIRE_ANGLE - diff;
            }
            if (score > bestScore) {
                bestScore = score;
                match = d;
                matchFace = face;
                matchSimilarity = similarity;
            }
        }

        float distance, personAngle;
        if (match >= 0) {
            const auto box = detector.getBox(match);
            if (!targetFound || countNotFound >= LOST_FRAMES) {
                std::cout << "[FollowPerson] " << (targetFound ? "found person again" : "found person, following");
                if (identify && matchSimilarity >= 0)
                    std::cout << " (" << (matchFace == PersonIdentifier::Face::Match ? "face" : "clothes")
                              << " similarity " << matchSimilarity << ")";
                std::cout << std::endl;
            }
            roamSearching = false;
            // their face is visible: remember how their clothes look on this camera
            if (matchFace == PersonIdentifier::Face::Match) identifier.learnClothing(detector, match);

            distance = boxDistance(box);
            personAngle = boxAngle(box);
            updatePersonTrack(rx + distance * std::cos(yaw + personAngle),
                              ry + distance * std::sin(yaw + personAngle),
                              targetFound && countNotFound == 0, now);
            lostSide = personAngle > 0 ? 1 : -1;

            targetFound = true;
            targetBox = box;
            countNotFound = 0;
        } else {
            if (!targetFound) continue;

            if (++countNotFound == LOST_FRAMES)
                std::cout << "[FollowPerson] person out of view, searching" << std::endl;

            // not where they were heading: walk around semi-randomly, the detector keeps
            // running every step, so the person is followed again as soon as they show up
            if (sinceSeen > SEARCH_TIMEOUT) {
                if (!roamSearching) {
                    std::cout << "[FollowPerson] person not found nearby, roaming around to find them" << std::endl;
                    explorer.reset();
                    roamSearching = true;
                }
                float x, z;
                explorer.step(x, z);
                drive(x, z);
                continue;
            }

            // head for the predicted position; it's outside the camera view, so turning
            // toward it means looking in the direction the person left
            distance = predDistance;
            personAngle = predAngle;
            if (std::abs(predAngle) > 0.1f) lostSide = predAngle > 0 ? 1 : -1;
        }

        // proportional control so the motion is smooth instead of walking straight and
        // suddenly turning once the person leaves a band around the image center
        // (x: move toward the person if too far, away if too close)
        float x = std::clamp(0.5f * (distance - TARGET_DISTANCE), -0.6f, +0.6f);

        // when walking toward the person, take a detour around obstacles in the way
        float heading = personAngle;
        if (x > 0) {
            float clearance = 0;
            heading = planHeading(personAngle, distance, clearance);
            // slow down in front of obstacles and only turn if nothing is free
            x = std::min(x, std::max(0.0f, 0.5f * (clearance - MIN_CLEARANCE)));
        }

        // z: rotate toward the chosen heading, proportional to how far off it is.
        // with this there is no "perfect center" needed for the rotation to start
        float z = std::clamp(TURN_GAIN * heading, -0.6f, +0.6f);

        // arrived where the person should be but still can't see them: look around,
        // starting with the side they disappeared to
        if (match < 0 && predDistance < SEARCH_ARRIVED) {
            x = 0;
            z = lostSide * SEARCH_TURN_SPEED;
        }

        drive(x, z);
    }
}

void FollowPerson::drive(float x, float z) {
    if (stairMode) {
        if (std::abs(roboControl.roll()) > STAIR_MAX_ROLL) {
            if (moving) std::cerr << "[FollowPerson] tilted sideways on stairs, stopping" << std::endl;
            stop();
            return;
        }

        // a nose-down body means the robot is going down the stairs
        const float maxSpeed = roboControl.pitch() > 0 ? STAIR_MAX_SPEED_DOWN : STAIR_MAX_SPEED_UP;
        x = std::clamp(x, 0.0f, maxSpeed);
        z = std::clamp(z, -STAIR_MAX_YAW, +STAIR_MAX_YAW);
    }

    if (x != 0 || z != 0) {
        roboControl.moveRelative(x, z);
        moving = true;
    } else {
        stop();
    }
}

void FollowPerson::updateStairMode(int& stairSamples, float& levelHeight) {
    const float pitch = std::abs(roboControl.pitch());
    const float height = roboControl.positionZ();

    if (!stairMode) {
        const bool tilted = pitch > STAIR_PITCH_ENTER;
        const bool heightChanged = std::abs(height - levelHeight) > STAIR_HEIGHT_CHANGE;
        stairSamples = (tilted || heightChanged) ? stairSamples + 1 : 0;

        if (stairSamples >= STAIR_ENTER_SAMPLES) {
            std::cout << "[FollowPerson] stairs detected, moving carefully" << std::endl;
            stairMode = true;
            stairSamples = 0;
        } else if (stairSamples == 0) {
            levelHeight = height;
        }
    } else {
        stairSamples = pitch < STAIR_PITCH_EXIT ? stairSamples + 1 : 0;

        if (stairSamples >= STAIR_EXIT_SAMPLES) {
            std::cout << "[FollowPerson] back on level ground" << std::endl;
            stairMode = false;
            stairSamples = 0;
            levelHeight = height;
        }
    }
}

void FollowPerson::stop() {
    // only send the stop once, repeating it would block the remote controller
    if (!moving) return;
    roboControl.stopMovement();
    moving = false;
}

void FollowPerson::updatePersonTrack(float x, float y, bool continuous, std::chrono::steady_clock::time_point now) {
    const float dt = std::chrono::duration<float>(now - lastSeen).count();
    if (continuous && dt > 0) {
        // smoothed, because the distance estimate from the box width is noisy
        personVX += VELOCITY_SMOOTHING * ((x - personX) / dt - personVX);
        personVY += VELOCITY_SMOOTHING * ((y - personY) / dt - personVY);
        const float speed = std::hypot(personVX, personVY);
        if (speed > MAX_PERSON_SPEED) {
            personVX *= MAX_PERSON_SPEED / speed;
            personVY *= MAX_PERSON_SPEED / speed;
        }
    } else {
        personVX = personVY = 0;
    }
    personX = x;
    personY = y;
    lastSeen = now;
}

float FollowPerson::planHeading(float personAngle, float distance, float& clearance) {
    const float needed = std::min(distance - PERSON_MARGIN, PLAN_LOOKAHEAD);
    if (!obstacleMap.available() || needed <= 0) {
        clearance = PLAN_LOOKAHEAD;
        return personAngle;
    }

    const float x = roboControl.positionX();
    const float y = roboControl.positionY();
    const float yaw = roboControl.yaw();

    // route through everything seen in the last minutes, so the robot can use a way
    // around that it saw earlier but that is out of the LiDAR's range now
    const float targetX = x + distance * std::cos(yaw + personAngle);
    const float targetY = y + distance * std::sin(yaw + personAngle);
    std::vector<std::pair<float, float>> path;
    if (obstacleMap.planPath(x, y, targetX, targetY, PERSON_MARGIN, path) && path.size() > 1) {
        auto aim = path.back();
        for (const auto& point : path) {
            if (std::hypot(point.first - x, point.second - y) >= PATH_LOOKAHEAD) {
                aim = point;
                break;
            }
        }
        const float heading = wrapAngle(std::atan2(aim.second - y, aim.first - x) - yaw);
        const float aimDistance = std::hypot(aim.first - x, aim.second - y);
        clearance = obstacleMap.freeDistance(x, y, yaw + heading, std::max(std::min(needed, aimDistance), MIN_CLEARANCE));
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
        const float heading = personAngle + k * DETOUR_STEP;
        const float free = obstacleMap.freeDistance(x, y, yaw + heading, needed);

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
