cmake_minimum_required(VERSION 3.10)
project(DirectMethodDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")

add_subdirectory(Modules/DirectMethod)

add_compile_options(-O3)

add_executable(demo example/main.cpp)
target_link_libraries(demo
    DirectMethod
)

add_executable(solution example/solution.cpp)
target_link_libraries(solution
    DirectMethod
)
