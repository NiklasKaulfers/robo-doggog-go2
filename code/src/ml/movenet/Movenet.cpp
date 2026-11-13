#include "Movenet.h"

#include <cmath>

#include "opencv2/opencv.hpp"

using namespace robo::sdk::ml::movenet;

cv::Point2f Movenet::toFrame(float x, float y) const {
    return {(x * EXPECTED_IMG_SIZE - framePadLeft) / frameScale, (y * EXPECTED_IMG_SIZE - framePadTop) / frameScale};
}

Keypoint Movenet::getKeypoint(int detection, int idx) const {
    auto& data = getKeypointData();
    int keypointCount = modelKeypointDataCount();
    return Keypoint{
        data[detection * keypointCount + idx * 3 + 0],
        data[detection * keypointCount + idx * 3 + 1],
        data[detection * keypointCount + idx * 3 + 2],
    };
}

bool Movenet::processInput(const std::vector<uint8_t> &data, litert::CompiledModel &model, litert::TensorBuffer &dest) {
    // data is a jpg image
    // https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_video_client.cpp

    lastFrame = cv::imdecode(data, cv::ImreadModes::IMREAD_COLOR);
    if (lastFrame.empty())
        return false;

    // fit the image into the square and pad the rest, portrait images (e.g. uploaded
    // photos) get padded left/right instead of top/bottom
    frameScale = std::min(static_cast<float>(EXPECTED_IMG_SIZE) / lastFrame.cols, static_cast<float>(EXPECTED_IMG_SIZE) / lastFrame.rows);
    const int width = std::min(EXPECTED_IMG_SIZE, static_cast<int>(std::lround(lastFrame.cols * frameScale)));
    const int height = std::min(EXPECTED_IMG_SIZE, static_cast<int>(std::lround(lastFrame.rows * frameScale)));
    cv::Mat img, tmp;
    cv::resize(lastFrame, tmp, cv::Size(width, height));
    framePadTop = (EXPECTED_IMG_SIZE - height) / 2;
    framePadLeft = (EXPECTED_IMG_SIZE - width) / 2;
    cv::copyMakeBorder(tmp, img, framePadTop, EXPECTED_IMG_SIZE - height - framePadTop, framePadLeft, EXPECTED_IMG_SIZE - width - framePadLeft, cv::BORDER_CONSTANT);
    if (tmp.type() != CV_8UC3) {
        img.convertTo(tmp, CV_8UC3);
        img = tmp;
    }
    cv::cvtColor(img, tmp, cv::COLOR_BGR2RGB);
    img = tmp;
    if (img.empty())
        return false;
    lastInput = img;
    assert(img.rows == EXPECTED_IMG_SIZE);
    assert(img.cols == EXPECTED_IMG_SIZE);

    // resize input tensor if needed
    LITERT_ASSIGN_OR_ABORT(const litert::RankedTensorType tensorType, dest.TensorType());
    if (tensorType.Layout().Dimensions()[1] != img.rows || tensorType.Layout().Dimensions()[2] != img.cols) {
        const auto status = model.ResizeInputTensor(0, litert::Span<const int>({1, img.rows, img.cols, 3}));
        if (!status.HasValue()) {
            std::cerr << "[Movenet] error while resizing input tensor: " << status.Error().Message() << std::endl;
            return false;
        }
        auto newTensor = model.CreateInputBuffers();
        if (!newTensor.HasValue()) {
            std::cerr << "[Movenet] error while resizing input tensor: " << newTensor.Error().Message() << std::endl;
            return false;
        }
        dest = std::move(newTensor.Value()[0]);
    }

    //TODO this is inefficient as fuck
    const int h = img.rows;
    const int w = img.cols;
    assert(inputDataBuf.size() == h * w * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const auto& pixel = img.at<cv::Vec3b>(y, x);
            int idx = (y * w + x) * 3;
            inputDataBuf[idx + 0] = pixel[0]; // R
            inputDataBuf[idx + 1] = pixel[1]; // G
            inputDataBuf[idx + 2] = pixel[2]; // B
        }
    }
    dest.Write<uint8_t>(absl::MakeConstSpan(inputDataBuf));

    return true;
}
