#include "TargetTracker.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "opencv2/imgproc.hpp"
#include "CameraGeometry.h"

using namespace robo::sdk::tracking;
using namespace robo::sdk::ml;

using Verdict = FaceRecognizer::Verdict;

// --- following people from frame to frame ---
// a detection continues a track when the boxes overlap this much, or the centers are
// this close (relative to the box size)
static constexpr float MATCH_IOU = 0.2f;
static constexpr float MATCH_CENTER_DISTANCE = 0.7f;
// tracks not seen for this many frames are dropped, so a person can be lost behind an
// obstacle for a moment without losing all the evidence about them
static constexpr int MAX_MISSES = 8;
// the target not seen for this many frames counts as lost
static constexpr int LOST_FRAMES = 3;
static constexpr float VELOCITY_SMOOTHING = 0.5f;
static constexpr float KEYPOINT_MIN_CONFIDENCE = 0.2f;

// --- face evidence ---
// how many people get their face checked per frame, the target always first
static constexpr int MAX_FACE_CHECKS = 3;
// evidence is -1..1: a match raises it toward 1 (a strong match all the way), a
// mismatch drops it halfway to -1. This much is needed to count as the person, this
// little to count as surely someone else
static constexpr float IDENTIFIED_EVIDENCE = 0.6f;
static constexpr float REJECTED_EVIDENCE = -0.5f;
// evidence fades slowly while the face isn't seen (per frame)
static constexpr float EVIDENCE_DECAY = 0.99f;
// another person's face evidence must beat the target's by this to take over
static constexpr float TAKEOVER_MARGIN = 0.3f;
// a face match this recent protects the target from being dropped by their clothes
static constexpr float RECENT_FACE_MATCH = 2.0f;
// how many faces are learned permanently from the camera when there are no photos
static constexpr int MAX_CAMERA_REFERENCES = 3;

// --- clothes ---
// while following: clothes this unlike the target's for this many frames in a row mean
// someone else is being followed
static constexpr float TRACK_MIN_SIMILARITY = 0.5f;
static constexpr int CLOTHING_LOW_FRAMES = 5;
// finding the target again by their clothes needs this much similarity (smoothed)
static constexpr float REACQUIRE_CLOTHING = 0.7f;
static constexpr float CLOTHING_SMOOTHING = 0.3f;
// new views of the clothes are only learned while they still look like the known ones
static constexpr float LEARN_CLOTHING_MIN = 0.6f;
// and only from a target followed for this many frames without interruption
static constexpr int STABLE_FRAMES = 10;
// another person overlapping the target this much makes their appearance unreliable
static constexpr float AMBIGUOUS_IOU = 0.05f;

// --- finding the target again by where they went ---
// for this long after losing them, a person showing up where they were heading is
// taken as them (within this angle), if nothing says otherwise
static constexpr float QUICK_REACQUIRE = 2.0f;
static constexpr float REACQUIRE_ANGLE = 0.4f;
// only people that showed up after the target was lost count for that
static constexpr float NEW_TRACK_SLACK = 0.5f;
// a target that has been seen for this many frames is taken without recognition when
// there is nothing to recognize them by
static constexpr int FIRST_PERSON_STABLE_FRAMES = 2;

// --- prediction ---
static constexpr float PREDICT_HORIZON = 2.0f;      // s, don't extrapolate further
static constexpr float MAX_PERSON_SPEED = 1.5f;     // m/s, caps noisy velocity estimates
static constexpr float WORLD_VELOCITY_SMOOTHING = 0.3f;

static constexpr auto STATUS_LOG_INTERVAL = std::chrono::seconds(5);

static float seconds(TargetTracker::Clock::duration d) {
    return std::chrono::duration<float>(d).count();
}

static std::string num(float value, int digits = 2) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(digits) << value;
    return s.str();
}

bool TargetTracker::init(const std::string& resourceDir) {
    return faces.init(resourceDir);
}

bool TargetTracker::loadReferencePhotos(const std::string& folder) {
    const int count = faces.enrollFolder(folder);
    if (count == 0) return false;
    referencePhotos = true;
    std::cout << "[Tracker] " << count << " reference photo(s) loaded, only the person in them is followed" << std::endl;
    return true;
}

void TargetTracker::dropTarget() {
    targetId = -1;
    haveWorldTrack = false;
    visibleLastFrame = false;
    current = Status();
}

TargetTracker::Track* TargetTracker::findTrack(int id) {
    if (id < 0) return nullptr;
    for (auto& t : tracks) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

float TargetTracker::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float intersection = (a & b).area();
    const float union_ = a.area() + b.area() - intersection;
    return union_ > 0 ? intersection / union_ : 0;
}

bool TargetTracker::identityKnown() const {
    return faces.gallerySize() > 0 || !clothes.empty();
}

bool TargetTracker::identifiedByFace(const Track& track) const {
    return track.faceEvidence >= IDENTIFIED_EVIDENCE;
}

bool TargetTracker::rejectedByFace(const Track& track) const {
    return track.faceEvidence <= REJECTED_EVIDENCE;
}

bool TargetTracker::unambiguous(const Track& track) const {
    for (const auto& other : tracks) {
        if (other.id == track.id || !other.matched) continue;
        if (iou(track.person.box, other.person.box) > AMBIGUOUS_IOU) return false;
        if (track.person.box.contains(other.person.center())) return false;
        if (other.person.box.contains(track.person.center())) return false;
    }
    return true;
}

cv::Rect2f TargetTracker::headRegion(const PersonDetection& person) {
    const int points[] = {PersonDetection::NOSE, PersonDetection::LEFT_EYE, PersonDetection::RIGHT_EYE,
                          PersonDetection::LEFT_EAR, PersonDetection::RIGHT_EAR};
    cv::Point2f center(0, 0);
    int count = 0;
    for (int k : points) {
        if (!person.has(k, KEYPOINT_MIN_CONFIDENCE)) continue;
        center += person.keypoints[k];
        count++;
    }
    // seen from behind there's no face to look for
    if (count == 0) return {};
    center *= 1.0f / count;

    // the head width from the ears, else the eyes, else a part of the body width
    float width;
    if (person.has(PersonDetection::LEFT_EAR, KEYPOINT_MIN_CONFIDENCE) && person.has(PersonDetection::RIGHT_EAR, KEYPOINT_MIN_CONFIDENCE)) {
        width = cv::norm(person.keypoints[PersonDetection::LEFT_EAR] - person.keypoints[PersonDetection::RIGHT_EAR]) * 1.4f;
    } else if (person.has(PersonDetection::LEFT_EYE, KEYPOINT_MIN_CONFIDENCE) && person.has(PersonDetection::RIGHT_EYE, KEYPOINT_MIN_CONFIDENCE)) {
        width = cv::norm(person.keypoints[PersonDetection::LEFT_EYE] - person.keypoints[PersonDetection::RIGHT_EYE]) * 3.0f;
    } else {
        width = person.box.width * 0.35f;
    }
    width = std::max(width, 16.0f);

    // the face reaches further down than the eyes/nose; some margin around helps the detector
    const float side = width * 1.7f;
    return cv::Rect2f(center.x - side / 2, center.y + width * 0.15f - side / 2, side, side);
}

const TargetTracker::Status& TargetTracker::update(const std::vector<PersonDetection>& detections, const cv::Mat& frame,
                                                   const RobotPose& pose, Clock::time_point now) {
    frameSize = frame.size();
    const float dt = haveTime ? std::max(seconds(now - lastUpdate), 1e-3f) : 0.1f;
    const float yawChange = havePose ? CameraGeometry::wrapAngle(pose.yaw - lastPose.yaw) : 0;

    predictTracks(dt, yawChange);
    associate(detections, now, dt);
    observeClothes(frame);
    observeFaces(frame, now);

    // where the target should be by now, for finding them again
    predictWorld(pose, now);
    visibleLastFrame = current.visible;
    chooseTarget(now);
    if (current.visible) {
        updateWorldTrack(*findTrack(targetId), pose, now);
        predictWorld(pose, now);
    }

    logStatus(findTrack(targetId), now);

    lastUpdate = now;
    lastPose = pose;
    haveTime = havePose = true;
    return current;
}

void TargetTracker::predictTracks(float dt, float yawChange) {
    const float shift = CameraGeometry::pixelShift(yawChange, frameSize.width);
    for (auto& t : tracks) {
        t.person.box.x += shift;
        t.previousCenter = t.person.center();
        t.person.box.x += t.velocity.x * dt;
        t.person.box.y += t.velocity.y * dt;
        t.matched = false;
        t.gapFrames = 0;
    }
}

void TargetTracker::associate(const std::vector<PersonDetection>& detections, Clock::time_point now, float dt) {
    struct Pair {
        float score;
        size_t track, detection;
    };
    std::vector<Pair> pairs;
    for (size_t i = 0; i < tracks.size(); i++) {
        const auto& box = tracks[i].person.box;
        const float size = std::max(box.width, box.height);
        for (size_t j = 0; j < detections.size(); j++) {
            const float overlap = iou(box, detections[j].box);
            const float distance = cv::norm(tracks[i].person.center() - detections[j].center()) / std::max(size, 1.0f);
            if (overlap < MATCH_IOU && distance > MATCH_CENTER_DISTANCE) continue;
            pairs.push_back({overlap + std::max(0.0f, 1 - distance), i, j});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.score > b.score; });

    std::vector<bool> detectionUsed(detections.size(), false);
    for (const auto& p : pairs) {
        Track& t = tracks[p.track];
        if (t.matched || detectionUsed[p.detection]) continue;
        const PersonDetection& d = detections[p.detection];

        const cv::Point2f measured = (d.center() - t.previousCenter) * (1.0f / dt);
        t.velocity = t.velocity * (1 - VELOCITY_SMOOTHING) + measured * VELOCITY_SMOOTHING;
        t.person = d;
        t.gapFrames = t.misses;
        t.misses = 0;
        t.hits++;
        t.lastSeen = now;
        t.matched = true;
        detectionUsed[p.detection] = true;
    }

    for (auto it = tracks.begin(); it != tracks.end();) {
        if (it->matched) {
            ++it;
            continue;
        }
        it->hits = 0;
        it->misses++;
        it->clothingSimilarity = -1;
        it->face = FaceRecognizer::Observation();
        it->faceChecked = false;
        if (it->misses > MAX_MISSES) it = tracks.erase(it);
        else ++it;
    }

    for (size_t j = 0; j < detections.size(); j++) {
        if (detectionUsed[j]) continue;
        Track t;
        t.id = nextTrackId++;
        t.person = detections[j];
        t.previousCenter = detections[j].center();
        t.hits = 1;
        t.firstSeen = t.lastSeen = now;
        t.matched = true;
        tracks.push_back(std::move(t));
    }
}

void TargetTracker::observeClothes(const cv::Mat& frame) {
    for (auto& t : tracks) {
        if (!t.matched) continue;
        if (!Appearance::describe(frame, t.person, t.signature)) {
            t.signature = Appearance::Signature();
            t.clothingSimilarity = -1;
            continue;
        }
        if (clothes.empty()) {
            t.clothingSimilarity = -1;
            continue;
        }
        t.clothingSimilarity = clothes.similarity(t.signature);
        if (t.clothingSimilarity < 0) continue;
        t.clothingAverage = t.clothingAverage < 0 ? t.clothingSimilarity
                                                  : t.clothingAverage + CLOTHING_SMOOTHING * (t.clothingSimilarity - t.clothingAverage);
        t.clothingLow = t.clothingSimilarity < TRACK_MIN_SIMILARITY ? t.clothingLow + 1 : 0;
    }
}

void TargetTracker::observeFaces(const cv::Mat& frame, Clock::time_point now) {
    // the target first, then the largest (closest) people
    std::vector<Track*> order;
    for (auto& t : tracks) {
        t.faceChecked = false;
        if (t.matched) order.push_back(&t);
    }
    std::sort(order.begin(), order.end(), [&](const Track* a, const Track* b) {
        if ((a->id == targetId) != (b->id == targetId)) return a->id == targetId;
        return a->person.box.area() > b->person.box.area();
    });

    int checks = 0;
    for (Track* t : order) {
        t->face = FaceRecognizer::Observation();
        if (checks >= MAX_FACE_CHECKS) break;
        const cv::Rect2f head = headRegion(t->person);
        if (head.empty()) continue;
        checks++;
        t->faceChecked = true;
        t->face = faces.observe(frame, head);
    }

    for (auto& t : tracks) {
        const auto& face = t.face;
        if (!t.matched || !face.found || face.similarity < 0) {
            t.faceEvidence *= EVIDENCE_DECAY;
            continue;
        }
        switch (face.verdict) {
            case Verdict::Match: {
                const float strength = std::clamp((face.similarity - FaceRecognizer::MATCH_THRESHOLD) /
                                                  (FaceRecognizer::STRONG_MATCH - FaceRecognizer::MATCH_THRESHOLD), 0.0f, 1.0f);
                t.faceEvidence += (1 - t.faceEvidence) * (0.5f + 0.5f * strength);
                t.lastFaceMatch = now;
                if (identifiedByFace(t)) t.verified = true;
                break;
            }
            case Verdict::Mismatch:
                t.faceEvidence -= (t.faceEvidence + 1) * 0.5f;
                break;
            case Verdict::Unknown:
                break;
        }
    }
}

void TargetTracker::chooseTarget(Clock::time_point now) {
    Track* target = findTrack(targetId);
    current.visible = false;

    if (target) {
        if (!target->matched) {
            // not seen this frame, the track waits for them a few frames
            current.state = target->misses >= LOST_FRAMES ? State::Lost : State::Tracking;
            if (current.state == State::Lost && !lostLogged) {
                std::cout << "[Tracker] person out of view, searching" << std::endl;
                lostLogged = true;
            }
            return;
        }

        const bool recentFace = seconds(now - target->lastFaceMatch) < RECENT_FACE_MATCH;
        const char* dropReason = nullptr;
        std::string details;
        if (rejectedByFace(*target)) {
            dropReason = "their face is not the one being looked for";
            details = "face similarity " + num(target->face.similarity);
        } else if (!clothes.empty() && target->clothingLow >= CLOTHING_LOW_FRAMES && !recentFace) {
            dropReason = "their clothes stopped looking like the target's";
            details = "clothing similarity " + num(target->clothingSimilarity);
        } else if (!clothes.empty() && target->gapFrames >= LOST_FRAMES && target->clothingSimilarity >= 0 &&
                   target->clothingSimilarity < TRACK_MIN_SIMILARITY && !recentFace) {
            dropReason = "someone with other clothes appeared where they disappeared";
            details = "clothing similarity " + num(target->clothingSimilarity);
        }
        if (dropReason) {
            std::cout << "[Tracker] dropping the person followed, " << dropReason << " (" << details << ")" << std::endl;
            // the continuity of the track is in doubt, so is the evidence it collected
            target->faceEvidence = 0;
            targetId = -1;
            target = nullptr;
            // where the wrong person went says nothing about the right one
            personVX = personVY = 0;
            current.state = State::Lost;
            lostLogged = true;
        }
    } else if (targetId >= 0) {
        // the track died
        targetId = -1;
        current.state = State::Lost;
        if (!lostLogged) {
            std::cout << "[Tracker] person out of view, searching" << std::endl;
            lostLogged = true;
        }
    }

    if (target) {
        // someone else whose face surely is the person: the tracker followed the wrong
        // one, e.g. after people crossed
        if (!(seconds(now - target->lastFaceMatch) < RECENT_FACE_MATCH)) {
            Track* better = nullptr;
            for (auto& t : tracks) {
                if (t.id == targetId || !t.matched || !identifiedByFace(t)) continue;
                if (t.faceEvidence <= target->faceEvidence + TAKEOVER_MARGIN) continue;
                if (!better || t.faceEvidence > better->faceEvidence) better = &t;
            }
            if (better) {
                std::cout << "[Tracker] the face of someone else matches better (" << num(better->face.similarity)
                          << "), switching to them" << std::endl;
                target = better;
                targetId = better->id;
            }
        }
    } else {
        // pick a new target
        Track* chosen = nullptr;
        std::string how;
        const float sinceSeen = haveWorldTrack ? seconds(now - lastSeen) : 1e9f;

        // 1. a face that matches
        for (auto& t : tracks) {
            if (!t.matched || !identifiedByFace(t)) continue;
            if (!chosen || t.faceEvidence > chosen->faceEvidence ||
                (t.faceEvidence == chosen->faceEvidence && t.person.box.area() > chosen->person.box.area())) chosen = &t;
        }
        if (chosen) how = "face similarity " + num(chosen->face.similarity);

        // 2. the clothes learned from them
        if (!chosen && !clothes.empty()) {
            for (auto& t : tracks) {
                if (!t.matched || t.hits < FIRST_PERSON_STABLE_FRAMES || rejectedByFace(t)) continue;
                if (t.clothingAverage < REACQUIRE_CLOTHING) continue;
                if (!chosen || t.clothingAverage > chosen->clothingAverage) chosen = &t;
            }
            if (chosen) how = "clothing similarity " + num(chosen->clothingAverage);
        }

        // 3. just lost: someone new showing up where they were heading
        if (!chosen && current.state == State::Lost && haveWorldTrack && sinceSeen < QUICK_REACQUIRE) {
            float bestDiff = REACQUIRE_ANGLE;
            for (auto& t : tracks) {
                if (!t.matched || rejectedByFace(t)) continue;
                if (seconds(lastSeen - t.firstSeen) > NEW_TRACK_SLACK) continue;
                if (t.clothingSimilarity >= 0 && t.clothingSimilarity < TRACK_MIN_SIMILARITY) continue;
                const float angle = CameraGeometry::angle(t.person.box, frameSize.width);
                const float diff = std::abs(CameraGeometry::wrapAngle(angle - current.predictedAngle));
                if (diff < bestDiff) {
                    bestDiff = diff;
                    chosen = &t;
                }
            }
            if (chosen) how = "showed up where they went";
        }

        // 4. nothing to recognize anyone by: the first person that stays in view
        if (!chosen && !identityKnown()) {
            for (auto& t : tracks) {
                if (!t.matched || t.hits < FIRST_PERSON_STABLE_FRAMES) continue;
                if (!chosen || t.person.score > chosen->person.score) chosen = &t;
            }
            if (chosen) {
                chosen->defining = true;
                how = "first person seen, learning their face and clothes";
            }
        }

        if (!chosen) {
            if (current.state == State::Tracking) current.state = State::Lost;
            return;
        }

        target = chosen;
        targetId = chosen->id;
        std::cout << "[Tracker] " << (haveWorldTrack ? "found the person again" : "found the person, following")
                  << " (" << how << ")" << std::endl;
    }

    current.state = State::Tracking;
    current.visible = true;
    current.target = target->person;
    current.faceSimilarity = target->face.found ? target->face.similarity : -1;
    current.clothingSimilarity = target->clothingSimilarity;
    current.verified = target->verified;
    lostLogged = false;
    learnFromTarget(*target, now);
}

void TargetTracker::learnFromTarget(Track& target, Clock::time_point now) {
    const bool trusted = target.verified || target.defining;
    const bool clear = unambiguous(target);
    const auto& face = target.face;
    const bool faceMatch = face.found && face.verdict == Verdict::Match;
    const bool goodFace = face.found && face.size >= FaceRecognizer::MIN_LEARN_FACE_SIZE &&
                          face.quality >= FaceRecognizer::MIN_LEARN_QUALITY;

    if (!faces.hasReferences() && cameraReferences < MAX_CAMERA_REFERENCES && goodFace && clear && trusted &&
        (faces.gallerySize() == 0 || face.similarity >= FaceRecognizer::MATCH_THRESHOLD)) {
        // no photos: the face of the person being followed becomes the identity
        faces.learn(face.feature, true);
        cameraReferences++;
        target.verified = true;
        target.lastFaceMatch = now;
        std::cout << "[Tracker] learned the person's face from the camera (" << cameraReferences << "/"
                  << MAX_CAMERA_REFERENCES << ")" << std::endl;
    } else if (faceMatch && clear) {
        faces.learn(face.feature, false);
    }

    if (target.signature.empty() || !clear) return;
    if (faceMatch) {
        clothes.learn(target.signature, true);
    } else if (trusted && target.hits >= STABLE_FRAMES &&
               (clothes.empty() || target.clothingSimilarity >= LEARN_CLOTHING_MIN)) {
        clothes.learn(target.signature, false);
    }
}

void TargetTracker::updateWorldTrack(const Track& target, const RobotPose& pose, Clock::time_point now) {
    const float distance = CameraGeometry::distance(target.person.box, frameSize.width);
    const float angle = CameraGeometry::angle(target.person.box, frameSize.width);
    const float x = pose.x + distance * std::cos(pose.yaw + angle);
    const float y = pose.y + distance * std::sin(pose.yaw + angle);

    const float dt = seconds(now - lastSeen);
    if (haveWorldTrack && visibleLastFrame && dt > 0) {
        // smoothed, because the distance estimate from the box width is noisy
        personVX += WORLD_VELOCITY_SMOOTHING * ((x - personX) / dt - personVX);
        personVY += WORLD_VELOCITY_SMOOTHING * ((y - personY) / dt - personVY);
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
    haveWorldTrack = true;
    current.lostSide = angle > 0 ? 1 : -1;
}

void TargetTracker::predictWorld(const RobotPose& pose, Clock::time_point now) {
    if (!haveWorldTrack) {
        current.predictedAngle = current.predictedDistance = current.secondsSinceSeen = 0;
        return;
    }
    const float sinceSeen = seconds(now - lastSeen);
    const float t = std::min(sinceSeen, PREDICT_HORIZON);
    const float predX = personX + personVX * t;
    const float predY = personY + personVY * t;
    current.predictedAngle = CameraGeometry::wrapAngle(std::atan2(predY - pose.y, predX - pose.x) - pose.yaw);
    current.predictedDistance = std::hypot(predX - pose.x, predY - pose.y);
    current.secondsSinceSeen = sinceSeen;
    if (!current.visible && std::abs(current.predictedAngle) > 0.1f) current.lostSide = current.predictedAngle > 0 ? 1 : -1;
}

void TargetTracker::logStatus(const Track* target, Clock::time_point now) {
    if (now - lastStatusLog < STATUS_LOG_INTERVAL) return;
    lastStatusLog = now;

    std::ostringstream s;
    s << "[Tracker] ";
    if (target && current.visible) {
        s << "following: distance " << num(CameraGeometry::distance(target->person.box, frameSize.width), 1) << "m";
        if (target->face.found) s << ", face " << num(target->face.similarity);
        else if (target->lastFaceMatch != Clock::time_point()) s << ", face matched " << num(seconds(now - target->lastFaceMatch), 0) << "s ago";
        if (target->clothingSimilarity >= 0) s << ", clothes " << num(target->clothingSimilarity);
    } else if (current.state == State::Lost) {
        s << "lost for " << num(current.secondsSinceSeen, 0) << "s, expected at " << num(current.predictedDistance, 1)
          << "m, " << num(current.predictedAngle * 180 / M_PI, 0) << " deg";
    } else {
        s << "waiting for the person";
    }
    s << " | people in view " << std::count_if(tracks.begin(), tracks.end(), [](const Track& t) { return t.matched; })
      << ", faces known " << faces.gallerySize() << ", clothing views " << clothes.size() << " (" << clothes.anchors() << " anchors)";
    std::cout << s.str() << std::endl;
}

void TargetTracker::drawDebug(cv::Mat& image) const {
    if (image.empty()) return;
    for (const auto& t : tracks) {
        if (!t.matched) continue;
        const bool isTarget = t.id == targetId;
        cv::Scalar color = isTarget ? cv::Scalar(0, 200, 0) : cv::Scalar(160, 160, 160);
        if (rejectedByFace(t)) color = cv::Scalar(0, 0, 220);
        else if (!isTarget && identifiedByFace(t)) color = cv::Scalar(0, 200, 200);

        cv::rectangle(image, t.person.box, color, isTarget ? 3 : 1);

        cv::Rect torso, legs;
        Appearance::regions(image, t.person, torso, legs);
        if (torso.area() > 0) cv::rectangle(image, torso, cv::Scalar(200, 120, 0), 1);
        if (legs.area() > 0) cv::rectangle(image, legs, cv::Scalar(200, 0, 120), 1);

        const cv::Rect2f head = headRegion(t.person);
        if (!head.empty()) cv::rectangle(image, head, cv::Scalar(255, 255, 0), 1);
        if (t.face.found) {
            const cv::Scalar faceColor = t.face.verdict == Verdict::Match ? cv::Scalar(0, 255, 0)
                                       : t.face.verdict == Verdict::Mismatch ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
            cv::rectangle(image, t.face.box, faceColor, 2);
        }

        std::ostringstream label;
        label << "#" << t.id << (isTarget ? " TARGET" : "") << (t.verified ? " verified" : "");
        std::ostringstream evidence;
        evidence << "face ev " << num(t.faceEvidence);
        if (t.face.found) evidence << " sim " << num(t.face.similarity);
        if (t.clothingSimilarity >= 0) evidence << " clothes " << num(t.clothingSimilarity);
        const cv::Point origin(std::max(0.0f, t.person.box.x), std::max(14.0f, t.person.box.y - 6));
        cv::putText(image, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(image, evidence.str(), origin + cv::Point(0, 14), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
    }

    std::ostringstream status;
    status << (current.state == State::Tracking ? "TRACKING" : current.state == State::Lost ? "LOST" : "SEARCHING");
    if (current.state == State::Lost)
        status << " " << num(current.secondsSinceSeen, 0) << "s, expected " << num(current.predictedAngle * 180 / M_PI, 0) << " deg";
    status << " | faces " << faces.gallerySize() << " clothes " << clothes.size();
    cv::putText(image, status.str(), cv::Point(10, image.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}
