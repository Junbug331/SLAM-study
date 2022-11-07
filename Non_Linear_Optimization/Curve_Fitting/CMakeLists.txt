cmake_minimum_required(VERSION 3.10)
project(curveFitting)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)

set(G2O_LIBS
    g2o_core 
    g2o_types_sba 
    g2o_types_slam3d
    g2o_solver_csparse 
    g2o_stuff 
    g2o_csparse_extension
    )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    )

add_compile_options(-pthread)

add_definitions(-DRES_DIR="${CMAKE_SOURCE_DIR}/res")

add_executable(gaussNewton src/gaussNewton.cpp)
target_link_libraries(gaussNewton PRIVATE
    ${OpenCV_LIBS}
    stdc++fs
    Eigen3::Eigen
)

add_executable(ceresCurveFitting src/ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting PRIVATE
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS}
    stdc++fs
    Eigen3::Eigen
)

add_executable(g2oCurveFitting src/g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting PRIVATE
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${G2O_LIBS}
    stdc++fs
    Eigen3::Eigen
)
