project(DirectMethod)
set(TARGET ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

#add_definitions(-DENABLE_SSE)
#add_compile_options(${SSE_FLAGS} -g -O3 -march=native -pthread)
add_compile_options(-g -pthread -O3)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Pangolin REQUIRED)


set(SRC_FILES 
    src/direct_method.cpp
)

add_library(${TARGET} ${SRC_FILES})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    Eigen3::Eigen
    spdlog::spdlog
    Sophus::Sophus
    stdc++fs
)