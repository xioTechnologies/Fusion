#include "Fusion.h"
#include <stdbool.h>
#include <stdio.h>

#define SAMPLE_PERIOD (0.01f) // replace with actual sample period

int main() {
    // Initialise structure
    FusionAhrs ahrs;
    FusionAhrsInitialise(&ahrs);

    // This loop should repeat for each new gyroscope measurement
    while (true) {
        // Read sensors (replace with actual sensor data)
        const FusionVector gyroscope = {0.0f, 0.0f, 0.0f};
        const FusionVector accelerometer = {0.0f, 0.0f, 1.0f};

        // Update AHRS algorithm
        FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, SAMPLE_PERIOD);

        // Print Euler angles
        const FusionEuler euler = FusionQuaternionToEuler(FusionAhrsGetQuaternion(&ahrs));

        printf("Roll %0.1f, Pitch %0.1f, Yaw %0.1f\n", euler.angle.roll, euler.angle.pitch, euler.angle.yaw);
    }
}
