#include "Singlepose.h"

using namespace robo::sdk::ml::movenet;

static constexpr int KEYPOINT_DATA_PER_DETECTION = 17 * 3;

static void checkInputTensor(const std::vector<litert::TensorBuffer>& tensors) {
    if (tensors.size() != 1)
        throw std::format_error("TF model input has more or less than 1 tensor");

    const auto& tensor = tensors[0];

    LITERT_ASSIGN_OR_ABORT(const litert::TensorBufferType bufferType, tensor.BufferType());
    LITERT_ASSIGN_OR_ABORT(const litert::RankedTensorType tensorType, tensor.TensorType());

    if (bufferType != litert::TensorBufferType::kHostMemory)
        throw std::format_error("TF model input tensor does not use host memory");

    if (tensorType.ElementType() != litert::ElementType::UInt8)
        throw std::format_error("TF model input tensor is not of int32 type");

    if (tensorType.Layout().Rank() != 4)
        throw std::format_error("TF model input tensor is not of rank 4");

    const auto& dims = tensorType.Layout().Dimensions();
    if (dims[0] != 1)
        throw std::format_error("TF model input tensor first dimension unexpected size");
    if (dims[1] != 256 || dims[2] != 256)
        throw std::format_error("TF model input tensor width or height unexpected size");
    if (dims[3] != 3)
        throw std::format_error("TF model input tensor channels unexpected size");
}

static void checkOutputTensor(const std::vector<litert::TensorBuffer>& tensors) {
    if (tensors.size() != 1)
        throw std::format_error("TF model output has more or less than 1 tensor");

    const auto& tensor = tensors[0];

    LITERT_ASSIGN_OR_ABORT(const litert::TensorBufferType bufferType, tensor.BufferType());
    LITERT_ASSIGN_OR_ABORT(const litert::RankedTensorType tensorType, tensor.TensorType());

    if (bufferType != litert::TensorBufferType::kHostMemory)
        throw std::format_error("TF model output tensor does not use host memory");

    if (tensorType.ElementType() != litert::ElementType::Float32)
        throw std::format_error("TF model output tensor is not of float32 type");

    if (tensorType.Layout().Rank() != 4)
        throw std::format_error("TF model output tensor is not of rank 4");

    const auto& dims = tensorType.Layout().Dimensions();
    if (dims[0] != 1)
        throw std::format_error("TF model output tensor first dimension unexpected size");
    if (dims[1] != 1)
        throw std::format_error("TF model output tensor second dimension unexpected size");
    if (dims[2] != 17)
        throw std::format_error("TF model output tensor keypoints unexpected count");
    if (dims[3] != 3)
        throw std::format_error("TF model output tensor keypoint-components unexpected count");
}

Singlepose::Singlepose() : model(), keypointData(KEYPOINT_DATA_PER_DETECTION) {}

bool Singlepose::init() {
    LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
    modelEnv = std::make_unique<litert::Environment>(std::move(env));

    auto modelLoad = litert::CompiledModel::Create(*modelEnv, "./res/movenet-singlepose.tflite", litert::HwAccelerators::kCpu);
    if (!modelLoad.HasValue()) {
        std::cerr << "[Movenet Singlepose] unable to load model: " << modelLoad.Error().Message() << std::endl;
        return false;
    }
    model = std::move(modelLoad.Value());

    LITERT_ASSIGN_OR_ABORT(auto inputBufs, model.CreateInputBuffers());
    LITERT_ASSIGN_OR_ABORT(auto outputBufs, model.CreateOutputBuffers());
    checkInputTensor(inputBufs);
    checkOutputTensor(outputBufs);
    this->inputBuffers = std::move(inputBufs);
    this->outputBuffers = std::move(outputBufs);

    return true;
}

bool Singlepose::run(const std::vector<uint8_t> &data) {
    if (!processInput(data, model, inputBuffers[0])) {
        std::cerr << "[Movenet Singlepose] error while converting or setting input image" << std::endl;
        return false;
    }

    auto status = model.Run(inputBuffers, outputBuffers);
    if (!status.HasValue()) {
        std::cerr << "[Movenet Singlepose] error while running inference: " << status.Error().Message() << std::endl;
        return false;
    }

    status = outputBuffers[0].Read(absl::MakeSpan(keypointData));
    if (!status.HasValue()) {
        std::cerr << "[Movenet Singlepose] error reading output tensor: " << status.Error().Message() << std::endl;
        return false;
    }

    return true;
}

int Singlepose::modelDetectionCount() const {
    return 1;
}

int Singlepose::modelKeypointDataCount() const {
    return KEYPOINT_DATA_PER_DETECTION;
}

const std::vector<float>& Singlepose::getKeypointData() const {
    return keypointData;
}
