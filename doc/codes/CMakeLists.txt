cmake_minimum_required(VERSION 3.21...3.26)

include(cmake/BuildType.cmake)

project(rand10-100 CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(TBB REQUIRED)

include(FetchContent)
find_package(ranx CONFIG
  COMPONENTS openmp
)
if(NOT ranx_FOUND)
  message(STATUS "Fetching ranx library...")
  FetchContent_Declare(
    ranx
    GIT_REPOSITORY https://github.com/arminms/ranx.git
    GIT_TAG main
  )
  # setting required ranx components
  set(RANX_COMPONENTS openmp
    CACHE STRING "Required components"
  )
  FetchContent_MakeAvailable(ranx)
endif()

set(PROGS
  rand_10-100_v1
  rand_10-100_v2
  rand_10-100_v3
  rand_10-100_v4
  rand_10-100_v5
  rand_10-100_v6
  rand_10-100_v7
  rand_10-100_v8
)

foreach(PROG ${PROGS})
  add_executable(${PROG} ${PROG}.cpp)
  # target_compile_features(${PROG} PRIVATE cxx_std_17)
  target_link_libraries(${PROG} PRIVATE TBB::tbb ranx::openmp)
endforeach()
