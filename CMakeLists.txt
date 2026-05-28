cmake_minimum_required(VERSION 3.14)
project(mpc_framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)


add_library(bicycle_model
    src/models/BicycleModel.cpp
)

target_include_directories(bicycle_model PUBLIC
    include
)

target_link_libraries(bicycle_model PUBLIC
    Eigen3::Eigen
)
