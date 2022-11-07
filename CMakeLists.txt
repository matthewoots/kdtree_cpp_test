cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_STANDARD 14)
project(kd)

# set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-deprecated-declarations")

set(BUILD_SHARED_LIBS ON)

find_package(Eigen3 REQUIRED)
include_directories(
    ${EIGEN3_INCLUDE_DIRS} 
    include
    kdtree
)

add_executable(${PROJECT_NAME}
    src/test.cpp
    kdtree/kdtree.c
)
target_include_directories (${PROJECT_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/kdtree)