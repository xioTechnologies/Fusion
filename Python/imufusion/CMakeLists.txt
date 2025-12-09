file(GLOB_RECURSE files "*.c")

add_library(imufusion-Python-Package ${files})

find_package(Python COMPONENTS NumPy)

target_include_directories(imufusion-Python-Package PRIVATE ${PYTHON_INCLUDE_PATH})

target_link_libraries(imufusion-Python-Package PRIVATE ${PYTHON_LIBRARIES} Python::NumPy)
