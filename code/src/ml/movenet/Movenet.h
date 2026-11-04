#ifndef FOLLOW_MOVENET_H
#define FOLLOW_MOVENET_H

#include <cstdint>
#include <vector>
#include <litert/cc/litert_compiled_model.h>

namespace robo::sdk::ml::movenet {

typedef struct {
    float x;
    float y;
    float confidence;
} Keypoint;

typedef struct {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
    float confidence;
} Box;

class Movenet {

public:
    static constexpr int KEYPOINT_NOSE = 0;
    static constexpr int KEYPOINT_LEFT_EYE = 1;
    static constexpr int KEYPOINT_RIGHT_EYE = 2;
    static constexpr int KEYPOINT_LEFT_EAR = 3;
    static constexpr int KEYPOINT_RIGHT_EAR = 4;
    static constexpr int KEYPOINT_LEFT_SHOULDER = 5;
    static constexpr int KEYPOINT_RIGHT_SHOULDER = 6;
    static constexpr int KEYPOINT_LEFT_ELBOW = 7;
    static constexpr int KEYPOINT_RIGHT_ELBOW = 8;
    static constexpr int KEYPOINT_LEFT_WRIST = 9;
    static constexpr int KEYPOINT_RIGHT_WRIST = 10;
    static constexpr int KEYPOINT_LEFT_HIP = 11;
    static constexpr int KEYPOINT_RIGHT_HIP = 12;
    static constexpr int KEYPOINT_LEFT_KNEE = 13;
    static constexpr int KEYPOINT_RIGHT_KNEE = 14;
    static constexpr int KEYPOINT_LEFT_ANKLE = 15;
    static constexpr int KEYPOINT_RIGHT_ANKLE = 16;

    Movenet() : inputDataBuf(EXPECTED_IMG_SIZE * EXPECTED_IMG_SIZE * 3) {}
    virtual ~Movenet() = default;

    virtual bool init() = 0;

    virtual bool run(const std::vector<uint8_t>& data) = 0;

    virtual int modelDetectionCount() const = 0;
    virtual int modelKeypointDataCount() const = 0;
    virtual const std::vector<float>& getKeypointData() const = 0;

    Keypoint getKeypoint(int detection, int idx) const;

protected:
    static constexpr int EXPECTED_IMG_SIZE = 256;

    bool processInput(const std::vector<uint8_t>& data, litert::CompiledModel& model, litert::TensorBuffer& dest);

private:
    std::vector<uint8_t> inputDataBuf;
};
}

#endif //FOLLOW_MOVENET_H
