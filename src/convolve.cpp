#include "convolve.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace {

void validate(const Image& src, const std::vector<float>& kernel,
              int ksize, int dimensions, int threads) {
    if (!src.valid() || ksize < 1 || ksize % 2 == 0 || threads < 1) {
        throw std::invalid_argument("invalid image, kernel size or thread count");
    }
    const auto side = static_cast<std::size_t>(ksize);
    if ((dimensions == 1 && kernel.size() != side) ||
        (dimensions == 2 && (kernel.size() / side != side || kernel.size() % side != 0)) ||
        !std::all_of(kernel.begin(), kernel.end(), [](float weight) {
            return std::isfinite(weight) && weight >= 0 && weight <= 1;
        })) {
        throw std::invalid_argument("invalid kernel coefficients or dimensions");
    }
}

template<class RowFunction>
void parallel_rows(int height, int requested, const RowFunction& rows) {
    const int count = std::min(height, requested);
    if (count == 1) {
        rows(0, height);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(count - 1);
    const int base = height / count;
    const int remainder = height % count;
    int start = 0;
    try {
        for (int worker = 0; worker < count - 1; ++worker) {
            const int end = start + base + (worker < remainder);
            workers.emplace_back(rows, start, end);
            start = end;
        }
        rows(start, height);
    } catch (...) {
        for (auto& worker : workers) {
            worker.join();
        }
        throw;
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

int edge(int position, int offset, int limit) {
    return position + std::clamp(offset, -position, limit - 1 - position);
}

std::uint8_t quantize(float value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 255.0f) + 0.5f);
}

void convolve_rows(const Image& src, Image& dst, const std::vector<float>& kernel,
                   int ksize, int begin, int end) {
    const int half = ksize / 2;
    for (int y = begin; y < end; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < src.channels; ++c) {
                float sum = 0;
                for (int ky = 0; ky < ksize; ++ky) {
                    const int sy = edge(y, ky - half, src.height);
                    const auto row = static_cast<std::size_t>(ky) * ksize;
                    for (int kx = 0; kx < ksize; ++kx) {
                        const int sx = edge(x, kx - half, src.width);
                        sum += src.at(sy, sx, c) * kernel[row + kx];
                    }
                }
                dst.at(y, x, c) = quantize(sum);
            }
        }
    }
}

void separable_passes(const Image& src, const std::vector<float>& kernel, int ksize,
                      int threads, Image& dst, std::vector<float>& tmp) {
    const int half = ksize / 2;
    parallel_rows(src.height, threads, [&](int begin, int end) {
        for (int y = begin; y < end; ++y) {
            for (int x = 0; x < src.width; ++x) {
                for (int c = 0; c < src.channels; ++c) {
                    float sum = 0;
                    for (int k = 0; k < ksize; ++k) {
                        sum += src.at(y, edge(x, k - half, src.width), c) * kernel[k];
                    }
                    tmp[src.index(y, x, c)] = sum;
                }
            }
        }
    });

    parallel_rows(src.height, threads, [&](int begin, int end) {
        for (int y = begin; y < end; ++y) {
            for (int x = 0; x < src.width; ++x) {
                for (int c = 0; c < src.channels; ++c) {
                    float sum = 0;
                    for (int k = 0; k < ksize; ++k) {
                        sum += tmp[src.index(edge(y, k - half, src.height), x, c)] * kernel[k];
                    }
                    dst.at(y, x, c) = quantize(sum);
                }
            }
        }
    });
}

} // namespace

Image convolve_naive(const Image& src, const std::vector<float>& kernel, int ksize) {
    return convolve_threaded(src, kernel, ksize, 1);
}

Image convolve_threaded(const Image& src, const std::vector<float>& kernel, int ksize, int threads) {
    validate(src, kernel, ksize, 2, threads);
    Image dst(src.width, src.height, src.channels);
    parallel_rows(src.height, threads, [&](int begin, int end) {
        convolve_rows(src, dst, kernel, ksize, begin, end);
    });
    return dst;
}

Image convolve_separable_naive(const Image& src, const std::vector<float>& kernel, int ksize) {
    return convolve_separable_threaded(src, kernel, ksize, 1);
}

Image convolve_separable_threaded(const Image& src, const std::vector<float>& kernel,
                                  int ksize, int threads) {
    validate(src, kernel, ksize, 1, threads);
    Image dst(src.width, src.height, src.channels);
    std::vector<float> tmp(src.data.size());
    separable_passes(src, kernel, ksize, threads, dst, tmp);
    return dst;
}

void convolve_separable_into(const Image& src, const std::vector<float>& kernel, int ksize,
                             int threads, Image& dst, std::vector<float>& intermediate) {
    validate(src, kernel, ksize, 1, threads);
    if (&src == &dst || !dst.valid() || dst.width != src.width ||
        dst.height != src.height || dst.channels != src.channels) {
        throw std::invalid_argument("destination must be distinct and match the source");
    }
    intermediate.resize(src.data.size());
    separable_passes(src, kernel, ksize, threads, dst, intermediate);
}
