include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(FusionSources
    FusionAhrs.c   FusionCompass.c
    FusionOffset.c
)

set(FusionHEADERS		
    Fusion.h           FusionAxes.h
    FusionCompass.h    FusionMath.h
    FusionAhrs.h       FusionCalibration.h
    FusionConvention.h FusionOffset.h
)

add_library(Fusion ${FusionSources})
add_library(Fusion::lib ALIAS Fusion)

target_include_directories(Fusion 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Fusion>
)

if(UNIX AND NOT APPLE)
    target_link_libraries(Fusion m) # link math library for Linux
endif()

set_target_properties(Fusion
    PROPERTIES
        PUBLIC_HEADER "${FusionHEADERS}"
        EXPORT_NAME lib
)

export(TARGETS Fusion
    NAMESPACE Fusion::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/FusionTargets.cmake
)

include(CMakePackageConfigHelpers)
configure_package_config_file(FusionConfig.cmake.in 
  ${CMAKE_CURRENT_BINARY_DIR}/FusionConfig.cmake
  NO_SET_AND_CHECK_MACRO
  PATH_VARS CMAKE_INSTALL_PREFIX
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Fusion"
)

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/FusionConfigVersion.cmake"
  COMPATIBILITY AnyNewerVersion
)

install(TARGETS Fusion
    EXPORT FusionTargets
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Fusion
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Fusion
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT FusionTargets
    FILE "FusionTargets.cmake"
    NAMESPACE Fusion::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Fusion"
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/FusionConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/FusionConfigVersion.cmake
    DESTINATION 
        ${CMAKE_INSTALL_LIBDIR}/cmake/Fusion
)
