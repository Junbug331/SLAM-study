cmake_minimum_required(VERSION 3.10)
project(denseMono)
set(TARGET denseMono)

set(CMAKE_CXX_STANDARD 14)

set(SRC_FILES
    src/dense_mapping.cpp
    src/utils.cpp
)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

add_library(${TARGET} ${SRC_FILES})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    Sophus::Sophus
    Eigen3::Eigen
    spdlog::spdlog
    stdc++fs
)

