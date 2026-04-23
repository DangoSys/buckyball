project(bb-tests)
cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0003 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BB_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(WORKLOADS_DIR ${BB_TESTS_DIR}/workloads)

add_subdirectory(workloads)
