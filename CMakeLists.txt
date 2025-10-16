cmake_minimum_required(VERSION 3.15)
project(Fusion
    VERSION 1.2.8
    LANGUAGES C
)

option(BUILD_EXAMPLES "Build examples" ON)

add_subdirectory("Fusion")
if(BUILD_EXAMPLES)
    add_subdirectory("Examples/Advanced")
    add_subdirectory("Examples/Simple")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("Python/Python-C-API") # do not include when run by CI
endif()
