cmake_minimum_required(VERSION 3.10)
project(ba-demo)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")

add_subdirectory(Modules/BundleAdjustment)
add_compile_options(-O3)
add_executable(demo example/main.cpp)

target_link_libraries(demo BundleAdjustment)