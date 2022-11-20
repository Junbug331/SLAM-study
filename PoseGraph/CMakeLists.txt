cmake_minimum_required(VERSION 3.10)
project(PoseGraphDemo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")

add_subdirectory(Modules/PoseGraph)

add_executable(demo example/main.cpp)

target_link_libraries(demo
    PoseGraph
)
