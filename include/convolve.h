#pragma once

#include "image.h"

Image convolve_naive(const Image& src, const std::vector<float>& kernel, int ksize);
Image convolve_threaded(const Image& src, const std::vector<float>& kernel, int ksize, int num_threads);
Image convolve_separable_naive(const Image& src, const std::vector<float>& kernel, int ksize);
Image convolve_separable_threaded(const Image& src, const std::vector<float>& kernel, int ksize, int num_threads);

// Output must be distinct from the input. Both buffers may be reused across calls.
void convolve_separable_into(const Image& src, const std::vector<float>& kernel, int ksize,
                             int num_threads, Image& dst, std::vector<float>& intermediate);
