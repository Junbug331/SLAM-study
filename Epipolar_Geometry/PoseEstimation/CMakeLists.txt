cmake_minimum_required(VERSION 3.10)
project(pose_estimation)
set(CMAKE_CXX_STANDARD 14)
set(TARGET ${PROJECT_NAME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DRES_DIR="${CMAKE_SOURCE_DIR}/res")

add_subdirectory(modules/PoseEstimation)

add_executable(${TARGET} example/main.cpp)
target_link_libraries(${TARGET} 
    PoseEstimation
    stdc++fs
)