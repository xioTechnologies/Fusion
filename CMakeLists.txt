cmake_minimum_required(VERSION 3.15)
project(Fusion)

option(FUSION_BUILD_EXAMPLES "Build examples" TRUE)
option(FUSION_BUILD_PYTHON_API "Build Python C API" TRUE) 

add_subdirectory("Fusion")
if (FUSION_BUILD_EXAMPLES)
    add_subdirectory("Examples/Advanced")
    add_subdirectory("Examples/Simple")
endif()
if(FUSION_BUILD_PYTHON_API)
    add_subdirectory("Python/Python-C-API") # do not include when run by CI
endif()
