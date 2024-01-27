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

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem iostreams regex)
find_package(gflags REQUIRED)
if (WIN32)
    find_package(glog REQUIRED)
endif()

add_definitions(-DOPENCV)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${Boost_LIBRARY_DIRS}
)

# 0:no-simd; 1:sse; 2:neon;
set(SIMD 0)
if(SIMD EQUAL 1)
    message(STATUS "use sse-Intel to compile")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4")
    add_definitions(-DUSE_SIMD)
elseif(SIMD EQUAL 2)
    message(STATUS "use neon to compile")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -flto -mfpu=neon")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -flto")
    add_definitions(-DUSE_SIMD)
    add_definitions(-DUSE_NEON)
elseif(SIMD EQUAL 0)
    message(STATUS "no-simd to compile")
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

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/binaries")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/binaries")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/binaries")

file(GLOB ALL_LIBRARY_HDRS
    ${PROJECT_SOURCE_DIR}/cftracker/common/*.hpp
    ${PROJECT_SOURCE_DIR}/cftracker/feature/*.hpp
    ${PROJECT_SOURCE_DIR}/cftracker/filters/*.hpp
    ${PROJECT_SOURCE_DIR}/cftracker/trackers/*.hpp
    ${PROJECT_SOURCE_DIR}/cftracker/trackers/eco/*.hpp
    ${PROJECT_SOURCE_DIR}/cftracker/trackers/kcf/*.hpp
)

file(GLOB ALL_LIBRARY_SRCS
    ${PROJECT_SOURCE_DIR}/cftracker/common/*.cpp
    ${PROJECT_SOURCE_DIR}/cftracker/feature/*.cpp
    ${PROJECT_SOURCE_DIR}/cftracker/filters/*.cpp
    ${PROJECT_SOURCE_DIR}/cftracker/trackers/*.cpp
    ${PROJECT_SOURCE_DIR}/cftracker/trackers/eco/*.cpp
    ${PROJECT_SOURCE_DIR}/cftracker/trackers/kcf/*.cpp
)

file(GLOB SIMD_LIBRARY_HDRS
    ${PROJECT_SOURCE_DIR}/cftracker/common/simd/*.hpp
    ${PROJECT_SOURCE_DIR}/cftracker/feature/simd/*.hpp
)

file(GLOB SIMD_LIBRARY_SRCS
    ${PROJECT_SOURCE_DIR}/cftracker/common/simd/*.cpp
    ${PROJECT_SOURCE_DIR}/cftracker/feature/simd/*.cpp
)

if(SIMD GREATER 0)
    list(APPEND ALL_LIBRARY_HDRS ${SIMD_LIBRARY_HDRS})
    list(APPEND ALL_LIBRARY_SRCS ${SIMD_LIBRARY_SRCS})
endif()

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install")
#set(INSTALL_LIB_DIR "${PROJECT_BINARY_DIR}/lib") # 假设安装目录在编译目录的lib子目录内
set(CMAKE_SKIP_BUILD_RPATH FALSE) # 编译时加上RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) # 编译时RPATH不使用安装的RPATH
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/binaries")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # 安装的执行文件加上RPATH

add_library(${PROJECT_NAME} SHARED ${ALL_LIBRARY_HDRS} ${ALL_LIBRARY_SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBRARIES})

add_executable(run_tracker.bin ${PROJECT_SOURCE_DIR}/examples/run_tracker.cpp)
target_link_libraries(run_tracker.bin PUBLIC ${PROJECT_NAME}
    ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} gflags glog)

install(
    TARGETS ${PROJECT_NAME} run_tracker.bin
    EXPORT CfTrackerExport
    ARCHIVE DESTINATION binaries
    LIBRARY DESTINATION binaries
    RUNTIME DESTINATION binaries
)

set(INSTALL_HDRS
    ${PROJECT_SOURCE_DIR}/cftracker/common/common.hpp
    ${PROJECT_SOURCE_DIR}/cftracker/trackers/base_tracker.hpp
)

foreach(HDR ${INSTALL_HDRS})
  file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR} ${HDR})
  get_filename_component(DIR ${REL_FIL} DIRECTORY)
  install(
    FILES ${HDR}
    DESTINATION include/${DIR}
  )
endforeach()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/configs DESTINATION .)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/datas DESTINATION .)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/models DESTINATION .)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/scripts DESTINATION .)

set(CFTRACKER_CMAKE_DIR share/cftracker/cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cftracker-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/cftracker/cftracker-config.cmake
  PATH_VARS CFTRACKER_CMAKE_DIR
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cftracker
)

install(
  EXPORT CfTrackerExport
  DESTINATION share/cftracker/cmake/
  FILE CfTrackerTargets.cmake
)

install(
  FILES ${PROJECT_BINARY_DIR}/cmake/cftracker/cftracker-config.cmake
  DESTINATION share/cftracker/
)

