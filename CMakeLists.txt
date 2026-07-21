cmake_minimum_required(VERSION 3.14)
project(mpc_framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(osqp)

set(osqp_DIR "${CMAKE_BINARY_DIR}/_deps/osqp-build")

FetchContent_Declare(
    osqp-eigen
    GIT_REPOSITORY https://github.com/robotology/osqp-eigen.git
    GIT_TAG        v0.11.0
)
FetchContent_MakeAvailable(osqp-eigen)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(mpc_py python/bindings/mpc_py.cpp)
target_link_libraries(mpc_py PRIVATE mpc_controller gp_residual_model)


add_library(bicycle_model
    src/models/BicycleModel.cpp
)

target_include_directories(bicycle_model PUBLIC
    include
)

target_link_libraries(bicycle_model PUBLIC
    Eigen3::Eigen
)

add_library(mpc_controller
    src/mpc/MPCController.cpp
    src/mpc/SMPCController.cpp
    src/mpc/SafetyFilter.cpp
)

target_include_directories(mpc_controller PUBLIC
    include
)

target_link_libraries(mpc_controller PUBLIC
    bicycle_model
    Eigen3::Eigen
    osqp
    OsqpEigen::OsqpEigen
)

add_library(gp_residual_model
    src/models/GPResidualModel.cpp
)
target_include_directories(gp_residual_model PUBLIC
    include
)
target_link_libraries(gp_residual_model PUBLIC
    bicycle_model
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

add_executable(tests
    tests/BicycleModelTest.cpp
    tests/MPCControllerTest.cpp
    tests/GPResidualModelTest.cpp
    tests/SMPCControllerTest.cpp
    tests/SafetyFilterTest.cpp
)
target_link_libraries(tests
    bicycle_model
    mpc_controller
    gp_residual_model
    GTest::gtest_main
)
target_compile_definitions(tests PRIVATE
    GP_PARAMS_PATH="${CMAKE_SOURCE_DIR}/data/gp_params.json"
)
include(GoogleTest)
gtest_discover_tests(tests)


add_executable(lane_keeping
    examples/lane_keeping/main.cpp
)
target_link_libraries(lane_keeping
    mpc_controller
)

add_executable(path_following
    examples/path_following/main.cpp
)
target_link_libraries(path_following
    mpc_controller
)

add_executable(comparison
    examples/comparison/main.cpp
)
target_link_libraries(comparison
    mpc_controller
    gp_residual_model
)
target_compile_definitions(comparison PRIVATE
    GP_PARAMS_PATH="${CMAKE_SOURCE_DIR}/data/gp_params.json"
)

add_executable(montecarlo
    examples/montecarlo/main.cpp
)
target_link_libraries(montecarlo
    mpc_controller
    gp_residual_model
)
target_compile_definitions(montecarlo PRIVATE
    GP_PARAMS_PATH="${CMAKE_SOURCE_DIR}/data/gp_params.json"
)

add_executable(safety_filter
    examples/safety_filter/main.cpp
)
target_link_libraries(safety_filter
    mpc_controller
    gp_residual_model
)
target_compile_definitions(safety_filter PRIVATE
    GP_PARAMS_PATH="${CMAKE_SOURCE_DIR}/data/gp_params.json"
)