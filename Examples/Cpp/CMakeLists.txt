add_executable(Cpp-Example main.cpp)

set_property(TARGET Cpp-Example PROPERTY CXX_STANDARD 20)

target_link_libraries(Cpp-Example PRIVATE Fusion)
