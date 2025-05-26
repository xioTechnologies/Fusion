file(GLOB_RECURSE files "*.c")

add_library(Fusion ${files})
target_include_directories(Fusion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(UNIX AND NOT APPLE)
    target_link_libraries(Fusion m) # link math library for Linux
endif()
