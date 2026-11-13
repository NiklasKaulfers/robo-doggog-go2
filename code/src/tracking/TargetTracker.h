#ifndef FOLLOW_TARGETTRACKER_H
#define FOLLOW_TARGETTRACKER_H

#include <chrono>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "../ml/Appearance.h"
#include "../ml/FaceRecognizer.h"
#include "../ml/PersonDetection.h"

namespace robo::sdk::tracking {

// position and heading of the robot in the odometry frame
struct RobotPose {
    float x = 0, y = 0, yaw = 0;
};

// Keeps track of the one person to follow among everyone the pose detector sees.
//
// Every person in view is tracked from frame to frame (by where their box moves), so
// the evidence about who they are can add up over time: whether their face matches the
// reference photos, and whether their clothes look like the ones learned from the
// target. The target is chosen and kept with these rules:
//
//  - with reference photos, only a person whose face matched is ever followed first.
//    Without photos, the first person seen is followed and their face and clothes are
//    learned from the camera, from then on they are the one being recognized
//  - while following, the person is kept by continuity of their box. Their clothes are
//    checked every frame (so someone stepping in front of them is noticed), their face
//    whenever it is visible. A face that clearly isn't theirs drops the target at once,
//    and a face match on someone else takes the target over
//  - when they are out of view, they are found again by their face, by their clothes,
//    or, right after losing them, by showing up where they were heading
class TargetTracker {

public:

    using Clock = std::chrono::steady_clock;

    enum class State {
        // no target yet (or dropped): waiting for the person to show up
        Searching,
        // the target is in view, or was a moment ago
        Tracking,
        // the target disappeared, their predicted position says where to look
        Lost,
    };

    struct Status {
        State state = State::Searching;
        // the target was seen in this frame, target is then valid
        bool visible = false;
        ml::PersonDetection target;
        // latest evidence about the target, -1 if none
        float faceSimilarity = -1;
        float clothingSimilarity = -1;
        // the face of the person being followed matched at least once
        bool verified = false;

        // where the target is expected (relative to the robot: angle positive = left),
        // from their last position and walking speed; valid when tracking or lost
        float predictedAngle = 0;
        float predictedDistance = 0;
        float secondsSinceSeen = 0;
        // 1 = left, -1 = right: the side where the person went out of view
        int lostSide = 1;
    };

    // resourceDir holds the face models
    bool init(const std::string& resourceDir = "./res");
    // enrolls the photos of the person to follow; false if no face was found in any
    bool loadReferencePhotos(const std::string& folder);
    bool hasReferencePhotos() const { return referencePhotos; }

    // processes a frame: detections are everyone found in frame (BGR, full resolution)
    const Status& update(const std::vector<ml::PersonDetection>& detections, const cv::Mat& frame,
                         const RobotPose& pose, Clock::time_point now);

    // forgets the target but not who they are: whoever is recognized next is followed
    void dropTarget();
    const Status& status() const { return current; }

    // draws the tracks and their evidence into image (the frame) for debugging
    void drawDebug(cv::Mat& image) const;

private:

    // one person followed from frame to frame
    struct Track {
        int id = 0;
        ml::PersonDetection person;
        // of the box center, frame pixels per second
        cv::Point2f velocity {0, 0};
        // where the box center was in the last frame, corrected for the robot's turning
        cv::Point2f previousCenter {0, 0};
        // consecutive frames seen / not seen
        int hits = 0;
        int misses = 0;
        // how many frames the track was not seen right before the current match
        int gapFrames = 0;
        Clock::time_point firstSeen;
        Clock::time_point lastSeen;
        bool matched = false;

        // -1..1, builds up with face matches and drops with mismatches
        float faceEvidence = 0;
        ml::FaceRecognizer::Observation face;
        bool faceChecked = false;
        Clock::time_point lastFaceMatch;
        // face matched at some point: this really is the person
        bool verified = false;
        // the target chosen without any way to recognize them (no photos, nothing
        // learned yet): the person's identity is defined by this track
        bool defining = false;

        ml::Appearance::Signature signature;
        float clothingSimilarity = -1;
        // smoothed over the last frames, -1 if unknown
        float clothingAverage = -1;
        // consecutive frames with clothes that don't look like the target's
        int clothingLow = 0;
    };

    ml::FaceRecognizer faces;
    ml::Appearance clothes;
    bool referencePhotos = false;
    // how many faces were learned permanently from the camera (without photos)
    int cameraReferences = 0;

    std::vector<Track> tracks;
    int nextTrackId = 1;
    int targetId = -1;
    Status current;
    cv::Size frameSize;

    bool havePose = false;
    RobotPose lastPose;
    Clock::time_point lastUpdate;
    bool haveTime = false;

    // world track of the target: last position and velocity in the odometry frame
    float personX = 0, personY = 0, personVX = 0, personVY = 0;
    bool haveWorldTrack = false;
    Clock::time_point lastSeen;
    bool visibleLastFrame = false;
    bool lostLogged = false;
    Clock::time_point lastStatusLog;

    void predictTracks(float dt, float yawChange);
    void associate(const std::vector<ml::PersonDetection>& detections, Clock::time_point now, float dt);
    void observeFaces(const cv::Mat& frame, Clock::time_point now);
    void observeClothes(const cv::Mat& frame);
    void chooseTarget(Clock::time_point now);
    void learnFromTarget(Track& target, Clock::time_point now);
    void updateWorldTrack(const Track& target, const RobotPose& pose, Clock::time_point now);
    void predictWorld(const RobotPose& pose, Clock::time_point now);
    void logStatus(const Track* target, Clock::time_point now);

    Track* findTrack(int id);
    // the face/head area of a person, empty if their head isn't visible
    static cv::Rect2f headRegion(const ml::PersonDetection& person);
    // whether no other person overlaps the track, so what is learned is really theirs
    bool unambiguous(const Track& track) const;
    bool identifiedByFace(const Track& track) const;
    bool rejectedByFace(const Track& track) const;
    // whether there is anything to recognize the person by
    bool identityKnown() const;
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
};
}

#endif //FOLLOW_TARGETTRACKER_H
