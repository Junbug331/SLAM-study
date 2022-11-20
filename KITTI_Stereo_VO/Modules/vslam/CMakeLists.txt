cmake_minimum_required(VERSION 3.10)
project(vslam)

set(CMAKE_CXX_STANDARD 14)
set(TARGET ${PROJECT_NAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

find_package(CSparse REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(g2o REQUIRED)
find_package(GTest REQUIRED)
find_package(GFlags REQUIRED)
find_package(glog REQUIRED)


set(G2O_LIBS
    g2o_core 
    g2o_types_sba 
    g2o_solver_csparse 
    g2o_stuff 
    g2o_csparse_extension
    )


add_library(${TARGET}
    src/frame.cpp
    src/config.cpp
    src/dataset.cpp
    src/feature.cpp
    src/frontend.cpp
    src/backend.cpp
    src/map.cpp
    src/mappoint.cpp
    src/viewer.cpp
    src/visual_odometry.cpp
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${GTest_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
)

target_link_libraries(${TARGET} PUBLIC
    ${Pangolin_LIBRARIES} GL GLU GLEW glut
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    ${GTEST_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${CSPARSE_LIBRARY}
    Eigen3::Eigen
    Sophus::Sophus
    glog::glog
    uuid
    tiff
    stdc++fs
    pthread
)




