cmake_minimum_required(VERSION 3.13)
project(my_package LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(_level_solver SHARED level_solver.cpp)
target_link_libraries(_level_solver PRIVATE pybind11::module)

set_target_properties(_level_solver PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}")
