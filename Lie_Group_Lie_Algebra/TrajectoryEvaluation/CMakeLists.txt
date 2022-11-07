cmake_minimum_required(VERSION 3.10)
project(trajectoryError)

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})

add_definitions(-DROOT_DIR="${CMAKE_SOURCE_DIR}")

add_executable(trajectoryError src/trajectoryError.cpp)
target_link_libraries(trajectoryError PRIVATE
    Eigen3::Eigen
    Sophus::Sophus
    ${Pangolin_LIBRARIES}
    stdc++fs
)

