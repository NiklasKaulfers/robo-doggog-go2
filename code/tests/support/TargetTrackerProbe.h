#ifndef FOLLOW_TESTS_TARGETTRACKERPROBE_H
#define FOLLOW_TESTS_TARGETTRACKERPROBE_H

// Looks into and sets the TargetTracker's per-person evidence. TargetTracker declares
// `friend struct TargetTrackerProbe` for this; it must be defined only here (one
// definition for all tests). Faces can't be drawn, so face evidence and observations
// are set directly.

#include <cmath>
#include <vector>
#include "opencv2/core.hpp"
#include "ml/FaceRecognizer.h"
#include "tracking/TargetTracker.h"

namespace robo::sdk::tracking {
struct TargetTrackerProbe {
    using Track = TargetTracker::Track;

    // the track whose box is horizontally closest to x
    static Track* nearest(TargetTracker& tracker, float x) {
        Track* best = nullptr;
        for (auto& t : tracker.tracks) {
            if (!best || std::abs(t.person.box.x - x) < std::abs(best->person.box.x - x)) best = &t;
        }
        return best;
    }
    static void setFaceEvidence(TargetTracker& tracker, float x, float evidence) {
        if (auto* t = nearest(tracker, x)) t->faceEvidence = evidence;
    }
    static float faceEvidence(TargetTracker& tracker, float x) {
        auto* t = nearest(tracker, x);
        return t ? t->faceEvidence : 0;
    }
    // as if a clear face of someone else had been seen on them in the last frame
    static void setMismatchFace(TargetTracker& tracker, float x, float similarity = 0.05f) {
        auto* t = nearest(tracker, x);
        if (!t) return;
        t->face = ml::FaceRecognizer::Observation();
        t->face.found = true;
        t->face.size = 60;
        t->face.compared = true;
        t->face.similarity = similarity;
        t->face.verdict = ml::FaceRecognizer::Verdict::Mismatch;
    }
    static std::vector<cv::Rect2f> boxes(const TargetTracker& tracker) {
        std::vector<cv::Rect2f> result;
        for (const auto& t : tracker.tracks) result.push_back(t.person.box);
        return result;
    }
    static size_t trackCount(const TargetTracker& tracker) { return tracker.tracks.size(); }
    static int targetId(const TargetTracker& tracker) { return tracker.targetId; }
    static const Track* target(const TargetTracker& tracker) {
        for (const auto& t : tracker.tracks)
            if (t.id == tracker.targetId) return &t;
        return nullptr;
    }
    // as if reference photos had been loaded: the gallery gets this face feature
    static void setReferenceFace(TargetTracker& tracker, const cv::Mat& feature) {
        tracker.faces.learn(feature, true);
        tracker.referencePhotos = true;
    }
    static size_t clothingViews(const TargetTracker& tracker) { return tracker.clothes.size(); }
    static size_t galleryFaces(const TargetTracker& tracker) { return tracker.faces.gallerySize(); }
    // how many tracks had their face looked for in the last frame
    static int facesChecked(const TargetTracker& tracker) {
        int count = 0;
        for (const auto& t : tracker.tracks) count += t.faceChecked ? 1 : 0;
        return count;
    }
    static bool faceChecked(TargetTracker& tracker, float x) {
        auto* t = nearest(tracker, x);
        return t && t->faceChecked;
    }
    // as if the face of this track matched at this time
    static void setLastFaceMatch(TargetTracker& tracker, float x, TargetTracker::Clock::time_point when) {
        if (auto* t = nearest(tracker, x)) t->lastFaceMatch = when;
    }
};
}

#endif //FOLLOW_TESTS_TARGETTRACKERPROBE_H
