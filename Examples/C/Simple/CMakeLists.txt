add_executable(Simple-Example main.c)

target_link_libraries(Simple-Example PRIVATE Fusion)

set(SENSOR_DATA_CSV "${CMAKE_SOURCE_DIR}/Examples/Sensor Data.csv")
add_definitions(-DSENSOR_DATA_CSV=\"${SENSOR_DATA_CSV}\")
