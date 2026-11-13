#ifndef FOLLOW_APPEARANCE_H
#define FOLLOW_APPEARANCE_H

#include <deque>
#include <vector>
#include "opencv2/core.hpp"
#include "PersonDetection.h"

namespace robo::sdk::ml {

// How a person's clothes look: color histograms of the torso and the legs, taken from
// the full resolution frame using the pose keypoints. The face only works when the
// person looks at the robot, the clothes recognize them from behind, which is most of
// the time when following someone.
//
// An Appearance holds the clothes of the one person being followed as a set of views
// (front, back, different lighting). The views learned while the face was recognized
// are anchors and never forgotten, the rest is a rolling set of recent views.
class Appearance {

public:

    struct Signature {
        // normalized color histograms, empty when the region was not visible
        std::vector<float> torso;
        std::vector<float> legs;

        bool empty() const { return torso.empty() && legs.empty(); }
    };

    // describes the clothes of a person in the frame (BGR). False if too little of the
    // person is visible to say anything
    static bool describe(const cv::Mat& frame, const PersonDetection& person, Signature& out);
    // 0..1 how alike two signatures are (1 = same clothes), -1 if they share no region
    static float compare(const Signature& a, const Signature& b);
    // the body regions describe() uses, in frame pixels, for debug drawings
    static void regions(const cv::Mat& frame, const PersonDetection& person, cv::Rect& torso, cv::Rect& legs);

    bool empty() const { return templates.empty(); }
    size_t size() const { return templates.size(); }
    size_t anchors() const;
    void clear() { templates.clear(); }

    // how much the signature looks like the learned clothes: the closest view counts.
    // -1 if nothing was learned yet or no region is shared
    float similarity(const Signature& signature) const;
    // remembers a view of the clothes; anchor views are the ones seen while the face
    // was recognized, those are never forgotten
    void learn(const Signature& signature, bool anchor);

private:

    struct Template {
        Signature signature;
        bool anchor;
    };

    std::deque<Template> templates;
};
}

#endif //FOLLOW_APPEARANCE_H
