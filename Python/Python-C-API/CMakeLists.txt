file(GLOB_RECURSE files "*.c")

add_library(imufusion ${files})

find_package(Python COMPONENTS NumPy)

target_include_directories(imufusion PRIVATE ${PYTHON_INCLUDE_PATH})

target_link_libraries(imufusion PRIVATE ${PYTHON_LIBRARIES} Python::NumPy)
