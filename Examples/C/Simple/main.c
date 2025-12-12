#include "Fusion.h"
#include <stdio.h>

#define SAMPLE_PERIOD (0.01f)

int main() {
    FusionAhrs ahrs;
    FusionAhrsInitialise(&ahrs);

    float seconds;
    FusionVector gyroscope;
    FusionVector accelerometer;
    float ignore;

    FILE *file = fopen(SENSOR_DATA_CSV, "r");

    char header[256];
    fgets(header, sizeof(header), file); // skip CSV header

    // Read and process each CSV line as if in real time
    while (fscanf(file, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                  &seconds,
                  &gyroscope.axis.x, &gyroscope.axis.y, &gyroscope.axis.z, &accelerometer.axis.x,
                  &accelerometer.axis.y, &accelerometer.axis.z,
                  &ignore, &ignore, &ignore) == 10) {
        FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, SAMPLE_PERIOD);

        const FusionEuler euler = FusionEulerFrom(FusionAhrsGetQuaternion(&ahrs));

        printf("seconds=%0.3f, roll=%0.1f, pitch=%0.1f, yaw=%0.1f\n", seconds, euler.angle.roll, euler.angle.pitch, euler.angle.yaw);
    }
}
