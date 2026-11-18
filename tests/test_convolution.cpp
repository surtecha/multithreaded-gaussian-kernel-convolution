#include "convolve.h"
#include "kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>

namespace {

void check(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<class Function>
void rejects(const Function& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected invalid_argument");
}

void test_kernels() {
    for (int size : {1, 3, 5, 11, 15}) {
        for (float sigma : {std::numeric_limits<float>::min(), 0.5f, 2.5f,
                            std::numeric_limits<float>::max()}) {
            const auto one = make_gaussian_kernel_1d(size, sigma);
            const auto two = make_gaussian_kernel(size, sigma);
            check(one.size() == static_cast<std::size_t>(size), "1D kernel dimensions");
            check(two.size() == static_cast<std::size_t>(size * size), "2D kernel dimensions");
            check(std::abs(std::accumulate(one.begin(), one.end(), 0.0) - 1) < 1e-6,
                  "1D kernel normalization");
            check(std::abs(std::accumulate(two.begin(), two.end(), 0.0) - 1) < 1e-6,
                  "2D kernel normalization");
            for (int y = 0; y < size; ++y) {
                check(std::isfinite(one[y]) && one[y] == one[size - 1 - y], "1D symmetry");
                for (int x = 0; x < size; ++x) {
                    const float weight = two[y * size + x];
                    check(std::isfinite(weight) && weight == two[y * size + size - 1 - x] &&
                          weight == two[(size - 1 - y) * size + x], "2D symmetry");
                }
            }
        }
    }

    for (int size : {-1, 0, 2, 4}) {
        rejects([=] { make_gaussian_kernel_1d(size, 1); });
        rejects([=] { make_gaussian_kernel(size, 1); });
    }
    for (float sigma : {-1.0f, 0.0f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        rejects([=] { make_gaussian_kernel_1d(3, sigma); });
        rejects([=] { make_gaussian_kernel(3, sigma); });
    }
}

Image reference(const Image& input, int size, double sigma) {
    Image result(input.width, input.height, input.channels);
    const int radius = size / 2;
    std::vector<double> weights;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            weights.push_back(std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)));
        }
    }
    const double scale = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int c = 0; c < input.channels; ++c) {
                double sum = 0;
                auto weight = weights.begin();
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int sy = std::clamp(y + dy, 0, input.height - 1);
                        const int sx = std::clamp(x + dx, 0, input.width - 1);
                        sum += input.at(sy, sx, c) * *weight++;
                    }
                }
                result.at(y, x, c) = static_cast<std::uint8_t>(std::lround(sum / scale));
            }
        }
    }
    return result;
}

void compare(const Image& expected, const Image& actual) {
    check(expected.width == actual.width && expected.height == actual.height &&
          expected.channels == actual.channels && actual.valid(), "output dimensions");
    int max_error = 0;
    double total_error = 0;
    for (std::size_t i = 0; i < expected.data.size(); ++i) {
        const int delta = std::abs(int(expected.data[i]) - int(actual.data[i]));
        max_error = std::max(max_error, delta);
        total_error += delta;
    }
    check(max_error <= 1, "maximum pixel error exceeds 1");
    check(total_error / expected.data.size() <= 0.05, "mean pixel error exceeds 0.05");
}

void test_convolution() {
    Image constant(3, 2, 3);
    for (std::size_t i = 0; i < constant.data.size(); ++i) {
        constexpr std::array<std::uint8_t, 3> color{0, 121, 255};
        constant.data[i] = color[i % color.size()];
    }
    Image impulse(19, 17, 1);
    impulse.at(8, 9, 0) = 255;
    Image boundary(2, 1, 1);
    boundary.data = {0, 100};
    Image gray(9, 5, 1);
    Image rgb(23, 13, 3);
    for (auto* input : {&gray, &rgb}) {
        for (std::size_t i = 0; i < input->data.size(); ++i) {
            input->data[i] = static_cast<std::uint8_t>((i * 31 + i / 7 * 17) % 256);
        }
    }
    Image single(1, 1, 4);
    single.data = {255, 17, 0, 38};

    for (const auto* input : {&constant, &impulse, &boundary, &gray, &rgb, &single}) {
        const auto original = input->data;
        for (int size : {1, 3, 5, 11, 15}) {
            const float sigma = size / 6.0f;
            const auto one = make_gaussian_kernel_1d(size, sigma);
            const auto two = make_gaussian_kernel(size, sigma);
            const auto expected = reference(*input, size, sigma);
            const auto direct = convolve_naive(*input, two, size);
            compare(expected, direct);
            if (input == &constant || input == &single || size == 1) {
                check(direct.data == original, "constant or identity convolution");
            }
            const auto separable = convolve_separable_naive(*input, one, size);
            compare(direct, separable);
            Image output(input->width, input->height, input->channels);
            std::vector<float> scratch{0};
            for (int threads : {1, 2, 4, 8, 32}) {
                const auto threaded = convolve_threaded(*input, two, size, threads);
                check(threaded.data == direct.data, "threaded direct equality");
                const auto combined = convolve_separable_threaded(*input, one, size, threads);
                check(combined.data == separable.data, "threaded separable equality");
                convolve_separable_into(*input, one, size, threads, output, scratch);
                check(output.data == separable.data, "reused buffers equality");
            }
        }
        check(input->data == original, "convolution modified the input");
    }

    const auto edges = convolve_separable_naive(boundary, {0.25, 0.5, 0.25}, 3);
    check(edges.data == std::vector<std::uint8_t>{25, 75}, "clamped boundary samples");
    const auto blurred = convolve_naive(impulse, make_gaussian_kernel(5, 1), 5);
    check(blurred.at(8, 9, 0) > blurred.at(8, 8, 0) &&
          blurred.at(8, 8, 0) == blurred.at(7, 9, 0) &&
          blurred.at(8, 8, 0) == blurred.at(8, 10, 0) &&
          blurred.at(8, 6, 0) == 0, "impulse symmetry and support");
}

void test_invalid_arguments() {
    Image image(5, 3, 1);
    Image wrong(3, 5, 1);
    std::vector<float> scratch;
    const auto one = make_gaussian_kernel_1d(3, 1);
    const auto two = make_gaussian_kernel(3, 1);
    for (int threads : {-1, 0}) {
        rejects([&] { convolve_threaded(image, two, 3, threads); });
        rejects([&] { convolve_separable_threaded(image, one, 3, threads); });
    }
    rejects([&] { convolve_naive(image, two, 2); });
    rejects([&] { convolve_naive(image, one, 3); });
    rejects([&] { convolve_separable_naive(image, two, 3); });
    rejects([&] { convolve_separable_naive(image, {std::numeric_limits<float>::quiet_NaN()}, 1); });
    rejects([&] { convolve_separable_into(image, one, 3, 1, image, scratch); });
    rejects([&] { convolve_separable_into(image, one, 3, 1, wrong, scratch); });
    rejects([&] { Image invalid(1, 1, 5); });
    image.data.clear();
    rejects([&] { convolve_naive(image, two, 3); });
}

} // namespace

int main() {
    try {
        test_kernels();
        test_convolution();
        test_invalid_arguments();
        std::cout << "Kernel, convolution and input checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
