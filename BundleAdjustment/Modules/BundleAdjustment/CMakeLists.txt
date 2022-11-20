cmake_minimum_required(VERSION 3.10)
project(BundleAdjustment)

set(TARGET ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
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

add_library(${TARGET} 
    src/bundle_adjustment.cpp
    src/BAL_handler.cpp
    src/SnavelyReprojection.cpp
)

add_compile_options(-O3)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${G2O_LIBS}
    cxsparse
    Sophus::Sophus
    Eigen3::Eigen
    spdlog::spdlog
    stdc++fs
)
