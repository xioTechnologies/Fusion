file(GLOB_RECURSE files "*.c")

add_library(Fusion ${files})

if(UNIX AND NOT APPLE)
    target_link_libraries(Fusion m) # link math library for Linux
endif()
