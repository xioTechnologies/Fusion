file(GLOB_RECURSE files "*.c")

add_library(imufusion-Python-Package ${files})

find_package(Python COMPONENTS Development NumPy)

target_link_libraries(imufusion-Python-Package PRIVATE Python::Python Python::NumPy)
