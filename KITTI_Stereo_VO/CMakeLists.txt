cmake_minimum_required(VERSION 3.10)
project(KITTI_VO_SLAM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++14 -O3 -fopenmp -pthread")

add_definitions(-DCONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")
add_definitions(-DKITTI_DIR="${CMAKE_CURRENT_SOURCE_DIR}/KITTI/dataset/sequences")

add_subdirectory(Modules/vslam)

add_executable(demo example/main.cpp)
target_link_libraries(demo vslam)


