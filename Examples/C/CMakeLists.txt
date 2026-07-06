add_subdirectory("Complete")
add_subdirectory("Minimal")

set(SENSOR_DATA_CSV "${CMAKE_SOURCE_DIR}/Examples/Sensor Data.csv")

target_compile_definitions(Examples-Complete PRIVATE SENSOR_DATA_CSV=\"${SENSOR_DATA_CSV}\")
target_compile_definitions(Examples-Minimal PRIVATE SENSOR_DATA_CSV=\"${SENSOR_DATA_CSV}\")
