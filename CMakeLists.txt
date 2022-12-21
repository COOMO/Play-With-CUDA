cmake_minimum_required(VERSION 3.10)

# add debug information
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_CFLAGS} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)

# Create project
set(ProjectName "play-with-cuda")
project(${ProjectName} LANGUAGES CXX CUDA)
# In all .cu src files, there is no need to manually add
# But for .cpp src files, if you need to include <cuda_runtime.h>, you must manually add
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

find_package(jetson-utils)
find_package(jetson-inference)

# add directory for libnvbuf-utils to program
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# build subdirectories
add_subdirectory(CUDAhelloWorld)