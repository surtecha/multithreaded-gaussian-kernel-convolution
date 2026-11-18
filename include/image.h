#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Image {
    int width, height, channels;
    std::vector<std::uint8_t> data;

    Image(int width, int height, int channels);

    static Image load(const std::string& path);
    void save(const std::string& path) const;

    std::size_t index(int y, int x, int c) const {
        return (static_cast<std::size_t>(y) * width + x) * channels + c;
    }

    std::uint8_t& at(int y, int x, int c) { return data[index(y, x, c)]; }
    const std::uint8_t& at(int y, int x, int c) const { return data[index(y, x, c)]; }
    bool valid() const;
};
