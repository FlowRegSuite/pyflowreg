cmake_minimum_required(VERSION 3.13)
project(pyflowreg LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

add_library(_level_solver MODULE level_solver.cpp)
target_link_libraries(_level_solve  r PRIVATE pybind11::module Python3::Python)

set_target_properties(_level_solver PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}"
)

install(TARGETS _level_solver
    LIBRARY DESTINATION pyflowreg
)
