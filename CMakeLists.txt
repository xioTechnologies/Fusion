cmake_minimum_required(VERSION 3.15)
project(Fusion)

add_subdirectory("Fusion")
add_subdirectory("Examples/C")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("Python/imufusion") # do not include when run by CI
endif ()
