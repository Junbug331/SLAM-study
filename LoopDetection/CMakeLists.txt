cmake_minimum_required(VERSION 3.10)
project(LoopDetection)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)


find_package(OpenCV REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(spdlog REQUIRED)
# set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
# set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${DBoW3_INCLUDE_DIRS}
)
add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")
add_compile_options(-O3)

add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training ${OpenCV_LIBS} ${DBoW3_LIBS} stdc++fs spdlog::spdlog)

add_executable(loop_closure loop_closure.cpp)
target_link_libraries(loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS} stdc++fs spdlog::spdlog)
