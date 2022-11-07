cmake_minimum_required(VERSION 3.10)
project(EigenGeometry)
set(CMAKE_CXX_STANDARD 14)

set(TARGET ${PROJECT_NAME})

find_package(Eigen3 REQUIRED)

add_executable(${TARGET} src/useGeometry.cpp)
target_link_libraries(${TARGET} PRIVATE
    Eigen3::Eigen
)
