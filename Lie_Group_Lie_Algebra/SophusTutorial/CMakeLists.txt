cmake_minimum_required(VERSION 3.10)
project(useSophus)

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

add_executable(useSophus src/useSophus.cpp)
target_link_libraries(useSophus PRIVATE
Eigen3::Eigen
Sophus::Sophus
)
