cmake_minimum_required(VERSION 3.10)
project(MonoReconstruction)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

add_subdirectory(Module/DenseMono)

add_executable(example example/main.cpp)
target_link_libraries(example
    denseMono
)

add_executable(solution example/solution.cpp)
target_link_libraries(solution
    denseMono
)
