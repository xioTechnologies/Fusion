cmake_minimum_required(VERSION 3.15)
project(Fusion)

add_subdirectory("Fusion")
add_subdirectory("Examples/Advanced")
add_subdirectory("Examples/Simple")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("Python/Python-C-API") # do not include when run by CI
endif()
