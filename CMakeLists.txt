cmake_minimum_required(VERSION 3.15)
project(Fusion)

# Add C++ compatibility option (default: OFF).
# Can be enabled by passing `-DFUSION_CXX_COMPATIBILITY_ENABLED=ON` to CMake.
option(FUSION_CXX_COMPATIBILITY_ENABLED "Enable C++ code compatibility" OFF)
add_definitions(-DFUSION_CXX_COMPATIBILITY=$<BOOL:${FUSION_CXX_COMPATIBILITY_ENABLED}>)

add_subdirectory("Fusion")
add_subdirectory("Examples/Advanced")
add_subdirectory("Examples/Simple")
add_subdirectory("Examples/CxxCompatibility" EXCLUDE_FROM_ALL)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("Python/Python-C-API") # do not include when run by CI
endif()
