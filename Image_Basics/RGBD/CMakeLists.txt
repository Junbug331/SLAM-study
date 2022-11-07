cmake_minimum_required(VERSION 3.10)
project(rgbd)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    )

add_compile_options(-pthread)

add_definitions(-DRES_DIR="${CMAKE_SOURCE_DIR}/res")

add_executable(${TARGET} src/jointMap.cpp)
target_link_libraries(${TARGET} PRIVATE
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    stdc++fs
    Eigen3::Eigen
    Sophus::Sophus
)