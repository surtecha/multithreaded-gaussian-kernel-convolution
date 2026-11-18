#include "image.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

std::size_t image_size(int width, int height, int channels) {
    if (width < 1 || height < 1 || channels < 1 || channels > 4 ||
        width > std::numeric_limits<int>::max() / channels) {
        return 0;
    }

    const auto stride = static_cast<std::size_t>(width) * channels;
    if (stride > std::numeric_limits<std::size_t>::max() / height) {
        return 0;
    }
    return stride * height;
}

} // namespace

Image::Image(int width, int height, int channels)
    : width(width), height(height), channels(channels),
      data(image_size(width, height, channels)) {
    if (data.empty()) {
        throw std::invalid_argument("invalid image dimensions or channel count");
    }
}

bool Image::valid() const {
    return !data.empty() && data.size() == image_size(width, height, channels);
}

Image Image::load(const std::string& path) {
    int width = 0, height = 0, channels = 0;
    const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw(
        stbi_load(path.c_str(), &width, &height, &channels, 0), stbi_image_free);
    if (!raw) {
        throw std::runtime_error("failed to load image: " + path);
    }
    Image image(width, height, channels);
    std::copy_n(raw.get(), image.data.size(), image.data.begin());
    return image;
}

void Image::save(const std::string& path) const {
    if (!valid() || !stbi_write_png(path.c_str(), width, height, channels,
                                   data.data(), width * channels)) {
        throw std::runtime_error("failed to write PNG: " + path);
    }
}
