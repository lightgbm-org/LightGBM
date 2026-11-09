cmake_minimum_required(VERSION 3.18)
project(LightGBMCAPITest LANGUAGES C CXX)

# Exercise an installed library, without inheriting build-tree include paths or
# dependencies from LightGBM's CMake targets.
set(LIGHTGBM_ROOT "" CACHE PATH "Directory containing the installed include/ and lib/")
option(LIGHTGBM_STATIC "Link to a static LightGBM build with OpenMP" OFF)
find_path(LIGHTGBM_INCLUDE_DIR LightGBM/c_api.h
    PATHS "${LIGHTGBM_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(LIGHTGBM_LIBRARY NAMES _lightgbm
    PATHS "${LIGHTGBM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(test_c_api test.c)
set_target_properties(test_c_api PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(test_c_api PRIVATE "${LIGHTGBM_INCLUDE_DIR}")
target_link_libraries(test_c_api PRIVATE "${LIGHTGBM_LIBRARY}")
if(LIGHTGBM_STATIC)
    # The source is still compiled as C; a static C++ library needs its C++ and
    # OpenMP runtimes when linking the final executable.
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    set_target_properties(test_c_api PROPERTIES LINKER_LANGUAGE CXX)
    find_library(NANOARROW_LIBRARY NAMES nanoarrow_static
        PATHS "${LIGHTGBM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(test_c_api PRIVATE "${NANOARROW_LIBRARY}" OpenMP::OpenMP_CXX)
endif()
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_c_api PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

enable_testing()
add_test(NAME c_api COMMAND test_c_api)
