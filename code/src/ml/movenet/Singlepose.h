#ifndef FOLLOW_SINGLEPOSE_H
#define FOLLOW_SINGLEPOSE_H

#include "Movenet.h"

namespace robo::sdk::ml::movenet {
class Singlepose : public Movenet {

    // https://www.kaggle.com/models/google/movenet/tfLite/singlepose-thunder-tflite-float16

public:
    Singlepose();
    ~Singlepose() override = default;

    bool init() override;

    bool run(const std::vector<uint8_t>& data) override;

    int modelDetectionCount() const override;
    int modelKeypointDataCount() const override;
    const std::vector<float>& getKeypointData() const override;

private:
    std::unique_ptr<litert::Environment> modelEnv;
    litert::CompiledModel model;
    std::vector<litert::TensorBuffer> inputBuffers;
    std::vector<litert::TensorBuffer> outputBuffers;
    std::vector<float> keypointData;
};
}

#endif //FOLLOW_SINGLEPOSE_H
