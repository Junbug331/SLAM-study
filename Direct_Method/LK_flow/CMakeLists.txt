cmake_minimum_required(VERSION 3.10)
project(LK-flow)
set(TARGET ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DENABLE_SSE)
add_definitions(-DRES_DIR="${CMAKE_SOURCE_DIR}/res")
add_compile_options(${SSE_FLAGS} -g -O3 -march=native -pthread)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

set(SRC_FILES 
    src/optical_flow.cpp
)

add_executable(${TARGET} ${SRC_FILES})
add_executable(solution src/solution.cpp)

target_link_libraries(${TARGET} 
    ${OpenCV_LIBS}
    Eigen3::Eigen
    spdlog::spdlog
    stdc++fs
)

target_link_libraries(solution 
    ${OpenCV_LIBS}
    Eigen3::Eigen
    spdlog::spdlog
    stdc++fs
)