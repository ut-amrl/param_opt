project(speedway)
cmake_minimum_required(VERSION 3.16)

message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Werror")

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  message(STATUS "Additional Flags for Release mode")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O2 -DNDEBUG")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  message(STATUS "Additional Flags for Debug mode")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")
endif()

message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(libs glog gflags boost_system lua5.1 amrl_shared_lib
    pthread Eigen3::Eigen ${OpenCV_LIBS})

add_subdirectory(src/shared)
include_directories(src/shared)
include_directories(src)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src/config_reader/include)

add_library(gnav
            src/gnav/ackermann_motion_primitives.cc
            src/gnav/constant_curvature_arcs.cc
            src/gnav/linear_evaluator.cc
            src/gnav/motion_primitives.cc) 
target_link_libraries(gnav ${libs})

add_executable(param_opt
			src/param_opt/param_opt_main.cpp)
target_link_libraries(param_opt ${libs} gnav)

add_executable(param_data_gen
			src/param_opt/param_data_gen.cpp)
target_link_libraries(param_data_gen ${libs} gnav)
