cmake_minimum_required( VERSION 3.10 )
project( visualizeGeometry )
set(CMAKE_CXX_STANDARD 14)


find_package( Eigen3 REQUIRED )
find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable( visualizeGeometry src/visualizeGeometry.cpp )
target_link_libraries( visualizeGeometry ${Pangolin_LIBRARIES} )
target_link_libraries( visualizeGeometry Eigen3::Eigen )