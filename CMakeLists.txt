cmake_minimum_required(VERSION 3.16)
project(gaussian_convolution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
include(CTest)

function(project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(gaussian STATIC src/convolve.cpp src/image.cpp)
target_include_directories(gaussian PUBLIC include PRIVATE third_party)
target_link_libraries(gaussian PRIVATE Threads::Threads)
project_warnings(gaussian)

add_library(stb OBJECT src/stb.cpp)
target_include_directories(stb PRIVATE third_party)
target_link_libraries(gaussian PRIVATE stb)
