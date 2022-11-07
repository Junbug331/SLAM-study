cmake_minimum_required(VERSION 3.10)
project(Undistort_Image)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DRES_DIR="${CMAKE_SOURCE_DIR}/res")

add_executable(${TARGET} src/undistortImage.cpp)
target_link_libraries(${TARGET} PRIVATE
    ${OpenCV_LIBS}
    stdc++fs
)