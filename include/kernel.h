#pragma once

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

inline std::vector<float> make_gaussian_kernel_1d(int size, float sigma) {
    if (size < 1 || size % 2 == 0 || !std::isfinite(sigma) || sigma <= 0) {
        throw std::invalid_argument("kernel size must be positive and odd, and sigma positive and finite");
    }

    std::vector<double> weights(size);
    for (int x = 0; x < size; ++x) {
        const double distance = static_cast<double>(x - size / 2) / sigma;
        weights[x] = std::exp(-0.5 * distance * distance);
    }
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::vector<float> kernel(size);
    for (int x = 0; x < size; ++x) {
        kernel[x] = static_cast<float>(weights[x] / sum);
    }
    return kernel;
}

inline std::vector<float> make_gaussian_kernel(int size, float sigma) {
    const auto one = make_gaussian_kernel_1d(size, sigma);
    const auto side = one.size();
    if (side > std::vector<float>{}.max_size() / side) {
        throw std::length_error("2D kernel is too large");
    }
    std::vector<float> kernel(side * side);
    for (std::size_t y = 0; y < side; ++y) {
        for (std::size_t x = 0; x < side; ++x) {
            kernel[y * side + x] = one[y] * one[x];
        }
    }
    return kernel;
}
