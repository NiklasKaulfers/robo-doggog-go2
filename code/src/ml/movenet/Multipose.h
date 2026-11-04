//
// Created by ubuntu on 7/27/26.
//

#ifndef FOLLOW_MULTIPOSE_H
#define FOLLOW_MULTIPOSE_H
#include "Movenet.h"

namespace robo::sdk::ml::movenet {
class Multipose : public Movenet {

    // https://www.kaggle.com/models/google/movenet/tfLite/multipose-lightning-tflite-float16

public:
    Multipose();
    ~Multipose() override = default;

    bool init() override;

    bool run(const std::vector<uint8_t>& data) override;

    int modelDetectionCount() const override;
    int modelKeypointDataCount() const override;
    const std::vector<float>& getKeypointData() const override;

    Box getBox(int detection) const;

private:
    std::unique_ptr<litert::Environment> modelEnv;
    litert::CompiledModel model;
    std::vector<litert::TensorBuffer> inputBuffers;
    std::vector<litert::TensorBuffer> outputBuffers;
    std::vector<float> keypointData;
};
}

#endif //FOLLOW_MULTIPOSE_H
