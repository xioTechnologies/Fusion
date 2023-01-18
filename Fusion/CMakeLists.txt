file(GLOB_RECURSE files "*.c")

add_library(Fusion ${files})
include_directories(.) # For the headers

if(UNIX AND NOT APPLE)
    target_link_libraries(Fusion m) # link math library for Linux
endif()
