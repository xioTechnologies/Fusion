cmake_minimum_required(VERSION 3.15)
project(Fusion)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


add_subdirectory("Fusion")
add_subdirectory("Examples/C/Advanced")
add_subdirectory("Examples/C/Simple")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory("Python/imufusion") # do not include when run by CI
endif()

configure_package_config_file(
    cmake/FusionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FusionConfig.cmake
    INSTALL_DESTINATION cmake/
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION cmake/
)
