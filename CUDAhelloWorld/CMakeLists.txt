file(GLOB Sources *.cpp *.cu)
file(GLOB Includes *.h )

add_executable(CUDAhelloWorld ${Sources})

target_link_libraries(CUDAhelloWorld jetson-inference ${CUDA_LIBRARIES})