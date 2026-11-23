#ifndef FAKE_LITERT_COMPILED_MODEL_H
#define FAKE_LITERT_COMPILED_MODEL_H

// Stub of the LiteRT C++ API, only what src/ml/movenet uses, so Movenet.cpp and
// Multipose.cpp compile and run unchanged in the simulation tests. A model "runs" by
// asking fakesdk::Bus for the output (the simulated people in MoveNet MultiPose layout);
// the letterboxing, tensor checks and output parsing are the real code.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>       // std::format_error, Multipose.cpp gets it through the real headers
#include <initializer_list>
#include <iostream>     // std::cerr, also reached through the real headers
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace absl {
template <typename T>
class Span {
public:
    Span(T* data, size_t size) : ptr(data), length(size) {}
    T* data() const { return ptr; }
    size_t size() const { return length; }

private:
    T* ptr;
    size_t length;
};

template <typename C>
Span<typename C::value_type> MakeSpan(C& container) {
    return {container.data(), container.size()};
}
template <typename C>
Span<const typename C::value_type> MakeConstSpan(const C& container) {
    return {container.data(), container.size()};
}
}

namespace litert {

// a list of values, constructible from a braced list like the real litert::Span
template <typename T>
class Span {
public:
    Span(std::initializer_list<std::remove_const_t<T>> list) : values(list) {}
    size_t size() const { return values.size(); }
    const std::remove_const_t<T>& operator[](size_t i) const { return values[i]; }
    const std::vector<std::remove_const_t<T>>& vector() const { return values; }

private:
    std::vector<std::remove_const_t<T>> values;
};

class Error {
public:
    explicit Error(std::string message = "") : message(std::move(message)) {}
    const std::string& Message() const { return message; }

private:
    std::string message;
};

template <typename T>
class Expected {
public:
    Expected(T value) : value(std::move(value)) {}
    Expected(litert::Error error) : error(std::move(error)) {}
    bool HasValue() const { return value.has_value(); }
    T& Value() { return *value; }
    const T& Value() const { return *value; }
    const litert::Error& Error() const { return error; }

private:
    std::optional<T> value;
    litert::Error error;
};

template <>
class Expected<void> {
public:
    Expected() = default;
    Expected(litert::Error error) : ok(false), error(std::move(error)) {}
    bool HasValue() const { return ok; }
    const litert::Error& Error() const { return error; }

private:
    bool ok = true;
    litert::Error error;
};

#define FAKE_LITERT_CONCAT_INNER(a, b) a##b
#define FAKE_LITERT_CONCAT(a, b) FAKE_LITERT_CONCAT_INNER(a, b)
// like the real macro: aborts when the expression failed, else assigns its value
#define LITERT_ASSIGN_OR_ABORT(decl, expr)                                                \
    auto FAKE_LITERT_CONCAT(fake_litert_expected_, __LINE__) = (expr);                   \
    if (!FAKE_LITERT_CONCAT(fake_litert_expected_, __LINE__).HasValue()) {               \
        std::cerr << "LITERT_ASSIGN_OR_ABORT: "                                            \
                  << FAKE_LITERT_CONCAT(fake_litert_expected_, __LINE__).Error().Message() \
                  << std::endl;                                                            \
        std::abort();                                                                      \
    }                                                                                      \
    decl = std::move(FAKE_LITERT_CONCAT(fake_litert_expected_, __LINE__).Value())

enum class ElementType { None, Bool, Int8, Int16, Int32, Int64, UInt8, Float16, Float32 };
enum class TensorBufferType { kUnknown, kHostMemory };
enum class HwAccelerators { kNone = 0, kCpu = 1, kGpu = 2, kNpu = 4 };

class Layout {
public:
    Layout() = default;
    explicit Layout(std::vector<int32_t> dims) : dims(std::move(dims)) {}
    size_t Rank() const { return dims.size(); }
    const std::vector<int32_t>& Dimensions() const { return dims; }

private:
    std::vector<int32_t> dims;
};

class RankedTensorType {
public:
    RankedTensorType() = default;
    RankedTensorType(litert::ElementType type, litert::Layout layout) : type(type), layout(std::move(layout)) {}
    litert::ElementType ElementType() const { return type; }
    const litert::Layout& Layout() const { return layout; }

private:
    litert::ElementType type = litert::ElementType::None;
    litert::Layout layout;
};

// host memory buffer; copies share the memory like handles to the same buffer
class TensorBuffer {
public:
    TensorBuffer() = default;
    TensorBuffer(RankedTensorType type, size_t bytes)
        : state(std::make_shared<State>(State {std::move(type), std::vector<uint8_t>(bytes)})) {}

    Expected<TensorBufferType> BufferType() const {
        if (!state) return litert::Error("empty tensor buffer");
        return TensorBufferType::kHostMemory;
    }
    Expected<RankedTensorType> TensorType() const {
        if (!state) return litert::Error("empty tensor buffer");
        return state->type;
    }
    template <typename T>
    Expected<void> Write(absl::Span<const T> data) {
        if (!state || data.size() * sizeof(T) != state->bytes.size())
            return litert::Error("write size does not match the tensor");
        std::memcpy(state->bytes.data(), data.data(), state->bytes.size());
        return {};
    }
    template <typename T>
    Expected<void> Read(absl::Span<T> data) const {
        if (!state || data.size() * sizeof(T) != state->bytes.size())
            return litert::Error("read size does not match the tensor");
        std::memcpy(data.data(), state->bytes.data(), state->bytes.size());
        return {};
    }

    // the raw bytes, for the stub's model
    std::vector<uint8_t>* bytes() const { return state ? &state->bytes : nullptr; }

private:
    struct State {
        RankedTensorType type;
        std::vector<uint8_t> bytes;
    };
    std::shared_ptr<State> state;
};

struct EnvironmentOption {
    int tag = 0;
};

class Environment {
public:
    static Expected<Environment> Create(const std::vector<EnvironmentOption>& options);
};

// a MoveNet MultiPose model: input uint8 [1, h, w, 3] (1 x 1 until resized, like the real
// model with a dynamic input), output float32 [1, 6, 56]
class CompiledModel {
public:
    CompiledModel() = default;

    // fails like the real one when the model file doesn't exist
    static Expected<CompiledModel> Create(Environment& env, const std::string& modelPath, HwAccelerators accelerators);

    Expected<std::vector<TensorBuffer>> CreateInputBuffers() const;
    Expected<std::vector<TensorBuffer>> CreateOutputBuffers() const;
    Expected<void> ResizeInputTensor(size_t index, const litert::Span<const int>& dims);
    // the output comes from fakesdk::Bus::runDetector
    Expected<void> Run(const std::vector<TensorBuffer>& inputs, std::vector<TensorBuffer>& outputs) const;

private:
    std::vector<int32_t> inputDims {1, 1, 1, 3};
    bool loaded = false;
};
}

#endif //FAKE_LITERT_COMPILED_MODEL_H
