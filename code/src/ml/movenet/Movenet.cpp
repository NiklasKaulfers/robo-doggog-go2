#include "Movenet.h"

#include "opencv2/opencv.hpp"

using namespace robo::sdk::ml::movenet;

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

    cv::Mat img = cv::imdecode(data, cv::ImreadModes::IMREAD_COLOR);
    if (img.empty())
        return false;

    int height = static_cast<int>(EXPECTED_IMG_SIZE / img.size().aspectRatio());
    cv::Mat tmp;
    cv::resize(img, tmp, cv::Size(EXPECTED_IMG_SIZE, height));
    int padTop = (EXPECTED_IMG_SIZE - height) / 2;
    int padBottom = padTop + (EXPECTED_IMG_SIZE - (height + 2 * padTop));
    cv::copyMakeBorder(tmp, img, padTop, padBottom, 0, 0, cv::BORDER_CONSTANT);
    if (tmp.type() != CV_8UC3) {
        img.convertTo(tmp, CV_8UC3);
        img = tmp;
    }
    cv::cvtColor(img, tmp, cv::COLOR_BGR2RGB);
    img = tmp;
    if (img.empty())
        return false;
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
