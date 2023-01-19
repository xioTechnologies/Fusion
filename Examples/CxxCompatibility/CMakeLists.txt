# !!! WARNING !!! This project force to use the C++ compiler. It can be only compiled when you pass the
# C++ compatibility flag `-DFUSION_CXX_COMPATIBILITY_ENABLED=ON` to the CMake command.
# This code example is the same as the `Simple-Example` project. It only tests the possibility to compile by C++ compiler.

set(APP_NAME "Simple-Example-CxxComp")
project(${APP_NAME} LANGUAGES CXX)

set(FILES "main.c")
add_executable(${APP_NAME} ${FILES})

set_property(TARGET ${APP_NAME} PROPERTY LANGUAGE CXX)
set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 11)
set_source_files_properties(${FILES} PROPERTIES LANGUAGE CXX)

target_link_libraries(${APP_NAME} PRIVATE Fusion)

target_include_directories("${APP_NAME}" PUBLIC
        $<TARGET_PROPERTY:Fusion,INCLUDE_DIRECTORIES>
)
