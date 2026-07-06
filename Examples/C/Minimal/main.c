#include "Fusion.h"
#include <stdbool.h>
#include <stdio.h>

int main() {
    // Configure AHRS algorithm
    FusionAhrs ahrs;
    FusionAhrsInitialise(&ahrs);

    FusionAhrsSettings settings = fusionAhrsDefaultSettings;
    settings.sampleRate = 100.0f; // Hz

    FusionAhrsSetSettings(&ahrs, &settings);

    // Open CSV file
    FILE *file = fopen(SENSOR_DATA_CSV, "r");

    char header[256];
    fgets(header, sizeof(header), file); // skip CSV header

    while (true) {
        // Read each CSV line as if reading sensor data in real-time
        float ignore;
        FusionVector gyroscope;
        FusionVector accelerometer;

        if (fscanf(file, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                   &ignore,
                   &gyroscope.axis.x, &gyroscope.axis.y, &gyroscope.axis.z,
                   &accelerometer.axis.x, &accelerometer.axis.y, &accelerometer.axis.z,
                   &ignore, &ignore, &ignore) != 10) {
            break;
        }

        // Update AHRS algorithm
        FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer);

        // Print Euler angles
        const FusionEuler euler = FusionQuaternionToEuler(FusionAhrsGetQuaternion(&ahrs));

        printf("Euler angles (degrees) = %0.1f, %0.1f, %0.1f\n",
               euler.angle.roll, euler.angle.pitch, euler.angle.yaw);
    }

    fclose(file);
    return 0;
}
