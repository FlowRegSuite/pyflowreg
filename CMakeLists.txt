cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0148 NEW)

project(pyflowreg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

pybind11_add_module(_level_solver pyflowreg/src/level_solver.cpp)

if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

install(TARGETS _level_solver
    LIBRARY DESTINATION src
)

install(DIRECTORY pyflowreg/util
    DESTINATION util
    FILES_MATCHING PATTERN "*.py"
)
