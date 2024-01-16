cmake_minimum_required(VERSION 2.8)
project(cftracker)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()
message(STATUS "compile type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS ON)
option(GPU            "use GPU"                 OFF)
option(OPENMP         "use OPENMP"              OFF)
option(CUDNN          "use CUDNN"               OFF)
option(MULTI_THREAD   "use MULTI_THREAD"        ON )

include_directories(${PROJECT_SOURCE_DIR})

message(STATUS "use opencv to compile")    
find_package(OpenCV REQUIRED)
add_definitions(-DOPENCV)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

#0:no-simd;1:sse-Intel;2:neon-TX2;
set(SIMD 1)
if(SIMD EQUAL 1)
    message(STATUS "use sse-Intel to compile")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4")
    add_definitions(-DUSE_SIMD)
elseif(SIMD EQUAL 2)
    message(STATUS "use neon to compile")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
    add_definitions(-DUSE_SIMD)
    add_definitions(-DUSE_NENOS)
elseif(SIMD EQUAL 0)
    message(STATUS "no-simd to compile")
endif()

if(MULTI_THREAD)
    add_definitions(-DUSE_MULTI_THREAD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

if(GPU)
    message(STATUS "use gpu to compile")
    find_package( CUDA QUIET REQUIRED )
    include(FindCUDA)
    add_definitions(-DGPU)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_53,code=sm_53
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62)
    set(CUDA_NVCC_FLGAS_DEBUG "-g -G -Og")
    set(CUDA_NVCC_FLGAS_RELEASE "-Ofast -lineinfo")
    include_directories( ${CUDA_INCLUDE_DIRS} )
    link_directories(/usr/local/cuda/lib64)
    set(LIBS "-lcuda -lcudart -lcublas -lcurand -lstdc++")
    if(CUDNN)
        message(STATUS "use cudnn to compile")
        add_definitions(-DCUDNN)
        set(LIBS ${LIBS} "-lcudnn")
    endif()
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPTIMAZE_FLAG} -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -pthread -Wall -lstdc++ -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

file(GLOB ALL_LIBRARY_HDRS
    ${PROJECT_SOURCE_DIR}/common/*.hpp
    ${PROJECT_SOURCE_DIR}/feature/*.hpp
    ${PROJECT_SOURCE_DIR}/filters/*.hpp
    ${PROJECT_SOURCE_DIR}/trackers/*.hpp
    ${PROJECT_SOURCE_DIR}/trackers/eco/*.hpp
)

file(GLOB ALL_LIBRARY_SRCS
    ${PROJECT_SOURCE_DIR}/common/*.cpp
    ${PROJECT_SOURCE_DIR}/feature/*.cpp
    ${PROJECT_SOURCE_DIR}/filters/*.cpp
    ${PROJECT_SOURCE_DIR}/trackers/*.cpp
    ${PROJECT_SOURCE_DIR}/trackers/eco/*.cpp
)

add_library(${PROJECT_NAME} SHARED ${ALL_LIBRARY_HDRS} ${ALL_LIBRARY_SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBRARIES})

