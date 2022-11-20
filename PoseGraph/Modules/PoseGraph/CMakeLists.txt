cmake_minimum_required(VERSION 3.10)
project(PoseGraph)
set(TARGET ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

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

set(SRC_FILES
    src/pose_graph_g2o_lie.cpp
)

# add_compile_options(-O2)
add_compile_options(-g)

add_library(${TARGET} ${SRC_FILES})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PUBLIC
    ${CERES_LIBRARIES}
    ${G2O_LIBS}
    cxsparse
    Sophus::Sophus
    Eigen3::Eigen
    spdlog::spdlog
    stdc++fs
)
