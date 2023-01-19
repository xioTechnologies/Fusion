file(GLOB_RECURSE files "*.c")

add_library(Fusion ${files})

include_directories(.)          # Propagate the headers to parent project.

if(UNIX AND NOT APPLE)
    target_link_libraries(Fusion m) # link math library for Linux
endif()
