# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Release build
set(CMAKE_BUILD_TYPE Release)

# Name and version are defined in "pyproject.toml" via scikit-build-core
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Define the module name as a variable
set(MODULE_NAME core)

# Set C++ standard 
set(CMAKE_CXX_STANDARD 20)

# Find the module development requirements
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# Find source files
file(GLOB_RECURSE SRC_FILES src/core/*.cpp)

# Add a library using FindPython's tooling
python_add_library(${MODULE_NAME} MODULE src/bindings.cpp ${SRC_FILES} WITH_SOABI)

# For simple headers include
target_include_directories(${MODULE_NAME} PUBLIC
        src/core/include
        src/core/external/nanoflann/
        src/core/external/eigen-3.4.0/
)

# Link libraries
target_link_libraries(${MODULE_NAME} PRIVATE pybind11::headers)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${MODULE_NAME} PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(${MODULE_NAME} PRIVATE USE_OPENMP)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# This is passing in the version as a define just as an example
target_compile_definitions(${MODULE_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS ${MODULE_NAME} DESTINATION bvh)
