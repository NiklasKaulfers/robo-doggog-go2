// The LiteRT stub's model: shapes like MoveNet MultiPose, inference output from fakesdk::Bus

#include "litert/cc/litert_compiled_model.h"

#include <filesystem>
#include <fakesdk/bus.hpp>

namespace litert {

static constexpr int DETECTIONS = 6;
static constexpr int VALUES_PER_DETECTION = 17 * 3 + 5;

static size_t elements(const std::vector<int32_t>& dims) {
    size_t count = 1;
    for (const int32_t d : dims) count *= static_cast<size_t>(d);
    return count;
}

Expected<Environment> Environment::Create(const std::vector<EnvironmentOption>&) {
    return Environment();
}

Expected<CompiledModel> CompiledModel::Create(Environment&, const std::string& modelPath, HwAccelerators) {
    if (!std::filesystem::exists(modelPath)) return Error("model file " + modelPath + " not found");
    CompiledModel model;
    model.loaded = true;
    return model;
}

Expected<std::vector<TensorBuffer>> CompiledModel::CreateInputBuffers() const {
    if (!loaded) return Error("model not loaded");
    return std::vector<TensorBuffer> {TensorBuffer(RankedTensorType(ElementType::UInt8, Layout(inputDims)), elements(inputDims))};
}

Expected<std::vector<TensorBuffer>> CompiledModel::CreateOutputBuffers() const {
    if (!loaded) return Error("model not loaded");
    const std::vector<int32_t> dims {1, DETECTIONS, VALUES_PER_DETECTION};
    return std::vector<TensorBuffer> {TensorBuffer(RankedTensorType(ElementType::Float32, Layout(dims)), elements(dims) * sizeof(float))};
}

Expected<void> CompiledModel::ResizeInputTensor(size_t index, const litert::Span<const int>& dims) {
    if (index != 0 || dims.size() != 4 || dims[0] != 1 || dims[3] != 3) return Error("unsupported input shape");
    inputDims.assign(dims.vector().begin(), dims.vector().end());
    return {};
}

Expected<void> CompiledModel::Run(const std::vector<TensorBuffer>& inputs, std::vector<TensorBuffer>& outputs) const {
    if (!loaded) return Error("model not loaded");
    if (inputs.size() != 1 || outputs.size() != 1) return Error("expected one input and one output tensor");
    if (!inputs[0].bytes() || inputs[0].bytes()->size() != elements(inputDims))
        return Error("input tensor does not match the model's input shape");
    std::vector<float> result(DETECTIONS * VALUES_PER_DETECTION, 0.0f);
    if (!fakesdk::bus().runDetector(result)) return Error("inference failed");
    auto* out = outputs[0].bytes();
    if (!out || result.size() * sizeof(float) != out->size()) return Error("output tensor has the wrong size");
    std::memcpy(out->data(), result.data(), out->size());
    return {};
}
}
