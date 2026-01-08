file(GLOB_RECURSE files "*.c")
file(GLOB_RECURSE header "*.h")

add_library(Fusion ${files})

target_include_directories(Fusion PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/>
)

set_property(TARGET Fusion PROPERTY CXX_STANDARD 20)

set_property(TARGET Fusion PROPERTY POSITION_INDEPENDENT_CODE TRUE)


if(UNIX AND NOT APPLE)
    target_link_libraries(Fusion m) # link math library for Linux
endif()

install(TARGETS Fusion
    EXPORT FusionTargets
)

install(FILES
    ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/include/
)
