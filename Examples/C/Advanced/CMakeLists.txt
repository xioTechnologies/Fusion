add_executable(Advanced-Example main.c)

target_link_libraries(Advanced-Example PRIVATE Fusion)

set(SENSOR_DATA_CSV "${CMAKE_SOURCE_DIR}/Examples/Sensor Data.csv")
add_definitions(-DSENSOR_DATA_CSV=\"${SENSOR_DATA_CSV}\")
