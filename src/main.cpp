#include "convolve.h"
#include "kernel.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

enum class Method { naive, separable, threaded, combined, reuse };

Method parse_method(std::string_view name) {
    if (name == "naive_2d") return Method::naive;
    if (name == "separable") return Method::separable;
    if (name == "threaded_2d") return Method::threaded;
    if (name == "separable_threaded") return Method::combined;
    if (name == "separable_reuse") return Method::reuse;
    throw std::invalid_argument("unknown method");
}

int positive_int(std::string_view text) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ptr != text.data() + text.size() || result.ec != std::errc{} || value < 1) {
        throw std::invalid_argument("expected a positive integer");
    }
    return value;
}

Image apply(Method method, const Image& input, const std::vector<float>& kernel,
            int size, int threads) {
    switch (method) {
    case Method::naive: return convolve_naive(input, kernel, size);
    case Method::separable: return convolve_separable_naive(input, kernel, size);
    case Method::threaded: return convolve_threaded(input, kernel, size, threads);
    case Method::combined: return convolve_separable_threaded(input, kernel, size, threads);
    case Method::reuse: {
        Image output(input.width, input.height, input.channels);
        std::vector<float> scratch;
        convolve_separable_into(input, kernel, size, threads, output, scratch);
        return output;
    }
    }
    throw std::logic_error("unknown method");
}

Image benchmark_image(int width, int height) {
    Image image(width, height, 3);
    std::uint32_t state = 0x124d9a35u;
    for (auto& sample : image.data) {
        state = 1664525u * state + 1013904223u;
        sample = static_cast<std::uint8_t>(state >> 24);
    }
    return image;
}

struct Difference {
    int max;
    double mean;
};

Difference difference(const Image& reference, const Image& actual) {
    int maximum = 0;
    double total = 0;
    for (std::size_t i = 0; i < reference.data.size(); ++i) {
        const int delta = std::abs(int(reference.data[i]) - int(actual.data[i]));
        maximum = std::max(maximum, delta);
        total += delta;
    }
    return {maximum, total / reference.data.size()};
}

struct Timing {
    double mean, median, minimum, stddev;
};

Timing summarize(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
    double variance = 0;
    for (double sample : samples) {
        const double delta = sample - mean;
        variance += delta * delta;
    }
    const double median = (samples[(n - 1) / 2] + samples[n / 2]) / 2;
    return {mean, median, samples.front(), std::sqrt(variance / n)};
}

struct Configuration {
    Method method;
    const char* name;
    int threads;
};

struct Trial {
    Trial(Configuration config, int runs) : configuration(config) { samples.reserve(runs); }

    Configuration configuration;
    std::vector<double> samples;
    Difference error{};
};

void benchmark_case(std::ofstream& csv, const Image& input, int size, int runs, int warmups) {
    const float sigma = size / 6.0f;
    const auto two = make_gaussian_kernel(size, sigma);
    const auto one = make_gaussian_kernel_1d(size, sigma);
    const auto reference = convolve_naive(input, two, size);

    std::vector<Trial> trials;
    trials.emplace_back(Configuration{Method::naive, "naive_2d", 1}, runs);
    trials.emplace_back(Configuration{Method::separable, "separable", 1}, runs);
    const unsigned logical = std::max(1u, std::thread::hardware_concurrency());
    for (int threads : {1, 2, 4, 8}) {
        if (threads > input.height || static_cast<unsigned>(threads) > logical) continue;
        trials.emplace_back(Configuration{Method::threaded, "threaded_2d", threads}, runs);
        trials.emplace_back(Configuration{Method::combined, "separable_threaded", threads}, runs);
        trials.emplace_back(Configuration{Method::reuse, "separable_reuse", threads}, runs);
    }

    Image reusable(input.width, input.height, input.channels);
    std::vector<float> scratch(input.data.size());
    // Rotate starting methods across rounds to distribute ordering effects.
    for (int round = -warmups; round < runs; ++round) {
        for (std::size_t i = 0; i < trials.size(); ++i) {
            auto& trial = trials[(i + static_cast<std::size_t>(round + warmups)) % trials.size()];
            const auto [method, name, threads] = trial.configuration;
            const auto& kernel = (method == Method::naive || method == Method::threaded) ? two : one;
            std::optional<Image> allocated;
            const auto start = Clock::now();
            if (method == Method::reuse) {
                convolve_separable_into(input, kernel, size, threads, reusable, scratch);
            } else {
                allocated.emplace(apply(method, input, kernel, size, threads));
            }
            const auto stop = Clock::now();
            if (round == -warmups) {
                trial.error = difference(reference, allocated ? *allocated : reusable);
                if (trial.error.max > 1 || trial.error.mean > 0.05) {
                    throw std::runtime_error(std::string("numerical tolerance exceeded by ") + name);
                }
            }
            if (round >= 0) {
                trial.samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            }
        }
    }

    const double baseline = summarize(trials.front().samples).median;
    const double megapixels = static_cast<double>(input.width) * input.height / 1e6;
    for (auto& trial : trials) {
        const auto [method, name, threads] = trial.configuration;
        const auto stats = summarize(trial.samples);
        csv << input.width << ',' << input.height << ',' << input.channels << ',' << size << ','
            << sigma << ',' << name << ',' << threads << ',' << warmups << ',' << runs << ','
            << (method == Method::reuse ? "excluded" : "included") << ','
            << stats.mean << ',' << stats.median << ',' << stats.minimum << ',' << stats.stddev << ','
            << megapixels * 1000 / stats.median << ',' << baseline / stats.median << ','
            << trial.error.max << ',' << trial.error.mean << '\n';
        if (!csv) throw std::runtime_error("failed to write benchmark CSV");
    }
}

void benchmark(const char* output, int runs, int warmups, std::string_view mode) {
    if (mode != "standard" && mode != "quick" && mode != "4k") {
        throw std::invalid_argument("benchmark mode must be standard, quick or 4k");
    }

    std::ofstream csv(output);
    if (!csv) throw std::runtime_error("failed to open benchmark CSV");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(8)
        << "width,height,channels,kernel_size,sigma,method,threads,warmups,runs,buffer_setup,"
           "mean_ms,median_ms,min_ms,stddev_ms,mp_per_second,speedup,max_abs_error,mean_abs_error\n";

    std::vector<std::pair<int, int>> sizes{{512, 512}};
    if (mode != "quick") {
        sizes.emplace_back(1024, 1024);
        sizes.emplace_back(1920, 1080);
    }
    if (mode == "4k") sizes.emplace_back(3840, 2160);
    for (const auto& [width, height] : sizes) {
        const auto image = benchmark_image(width, height);
        for (int size : {3, 5, 11, 15}) {
            benchmark_case(csv, image, size, runs, warmups);
            std::cout << width << 'x' << height << ", kernel " << size << " complete" << std::endl;
        }
    }
    csv.close();
}

void usage() {
    std::cerr
        << "Usage:\n"
        << "  gaussian_bench blur INPUT OUTPUT.png KERNEL [SIGMA [THREADS [METHOD]]]\n"
        << "  gaussian_bench benchmark CSV [RUNS [WARMUPS [standard|quick|4k]]]\n"
        << "Methods: naive_2d, separable, threaded_2d, separable_threaded, separable_reuse\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string_view command = argc > 1 ? argv[1] : "";
        if (command == "blur" && argc >= 5 && argc <= 8) {
            const int size = positive_int(argv[4]);
            const std::string sigma_arg = argc > 5 ? argv[5] : "";
            std::size_t consumed = 0;
            const float sigma = argc > 5 ? std::stof(sigma_arg, &consumed) : size / 6.0f;
            if (consumed != sigma_arg.size()) throw std::invalid_argument("invalid sigma");
            const int threads = argc > 6 ? positive_int(argv[6]) : 1;
            const auto method = argc > 7 ? parse_method(argv[7]) : Method::combined;
            if ((method == Method::naive || method == Method::separable) && threads != 1) {
                throw std::invalid_argument("single-threaded methods require one thread");
            }
            const auto kernel = (method == Method::naive || method == Method::threaded)
                ? make_gaussian_kernel(size, sigma) : make_gaussian_kernel_1d(size, sigma);
            apply(method, Image::load(argv[2]), kernel, size, threads).save(argv[3]);
            return 0;
        }
        if (command == "benchmark" && argc >= 3 && argc <= 6) {
            const int runs = argc > 3 ? positive_int(argv[3]) : 5;
            const int warmups = argc > 4 ? positive_int(argv[4]) : 2;
            if (warmups > std::numeric_limits<int>::max() - runs) {
                throw std::invalid_argument("too many benchmark rounds");
            }
            benchmark(argv[2], runs, warmups, argc > 5 ? argv[5] : "standard");
            return 0;
        }
        usage();
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
