cmake_minimum_required(VERSION 3.21...3.26)

project(rand100 CXX)

## include necessary modules
#
include(FetchContent)
include(CheckLanguage)

## find ranx, if not installed fetch it...
#
find_package(ranx CONFIG
  COMPONENTS openmp
  OPTIONAL_COMPONENTS cuda oneapi rocm
)
if(NOT ranx_FOUND)
  message(STATUS "Fetching ranx library...")
  FetchContent_Declare(
    ranx
    GIT_REPOSITORY https://github.com/arminms/ranx.git
    GIT_TAG main
  )
  # setting required ranx components
  set(RANX_COMPONENTS openmp cuda oneapi rocm
    CACHE STRING "Required components"
  )
  FetchContent_MakeAvailable(ranx)
endif()

## build CUDA version if it's available
#
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_executable(rand100_cuda rand100.cu)
  target_link_libraries(rand100_cuda PRIVATE ranx::cuda)
endif()

## build ROCm version if it's available
#
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  add_executable(rand100_rocm rand100_rocm.cpp)
  target_link_libraries(rand100_rocm PRIVATE ranx::rocm)
endif()

## build oneAPI version if it's available, go with OpenMP otherwise
#
find_package(IntelDPCPP CONFIG)
if (IntelDPCPP_FOUND)
  add_executable(rand100_oneapi rand100_oneapi.cpp)
  target_link_libraries(rand100_oneapi PRIVATE ranx::oneapi)
else() # openmp and oneapi are mutually exclusive
  add_executable(rand100_openmp rand100_openmp.cpp)
  target_link_libraries(rand100_openmp PRIVATE ranx::openmp)
endif()
