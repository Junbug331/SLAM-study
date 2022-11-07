cmake_minimum_required(VERSION 3.10)
project(PlotTrajectory)
set(CMAKE_CXX_STANDARD 14)

set(TARGET ${PROJECT_NAME})

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

add_definitions(-DROOT_DIR="${CMAKE_SOURCE_DIR}")

include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(${TARGET} src/plotTrajectory.cpp)
target_link_libraries(${TARGET} PRIVATE
    Eigen3::Eigen
    ${Pangolin_LIBRARIES}
    stdc++fs
)