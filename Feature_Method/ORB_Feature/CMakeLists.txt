cmake_minimum_required(VERSION 3.10)
project(ORB-fast)
set(TARGET ${PROJECT_NAME})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DRES_DIR="${CMAKE_SOURCE_DIR}/res")
add_definitions(-DENABLE_SSE)
add_compile_options(-pthread -O2 ${SSE_FLAGS} -msse4)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${TARGET} src/fast_corner.cpp)

target_link_libraries(${TARGET} PRIVATE 
    ${OpenCV_LIBS}
    stdc++fs
    spdlog::spdlog
)