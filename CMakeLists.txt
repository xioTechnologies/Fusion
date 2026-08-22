cmake_minimum_required(VERSION 3.15)
project(Fusion)

include(GNUInstallDirs)

add_subdirectory("Fusion")
add_subdirectory("Examples/C/Advanced")
add_subdirectory("Examples/C/Simple")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("Python/imufusion") # do not include when run by CI
endif()
