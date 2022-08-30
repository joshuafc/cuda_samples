cmake_minimum_required(VERSION 3.22)
project(cudaTest CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep")
set(CMAKE_CUDA_ARCHITECTURES 75)
add_executable(cudaTest main.cu)

target_include_directories(cudaTest PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cudaTest ${OpenCV_LIBS} cublas)
set_target_properties(cudaTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
