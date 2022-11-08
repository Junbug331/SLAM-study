cmake_minimum_required(VERSION 3.10)
project(PoseEstimation)
set(TARGET ${PROJECT_NAME})
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)

set(G2O_LIBS
    g2o_core 
    g2o_types_sba 
    g2o_types_slam3d
    g2o_solver_csparse 
    g2o_stuff 
    g2o_csparse_extension
    )


set(SRC_FILE
    src/pose_estimation.cpp
    src/utils.cpp
    src/triangulation.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

add_compile_options(-pthread)
add_library(${TARGET} ${SRC_FILE})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${G2O_LIBS}
    Sophus::Sophus
    Eigen3::Eigen
    spdlog::spdlog
)



