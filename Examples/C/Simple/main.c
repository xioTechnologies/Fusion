#include "Fusion.h"
#include <stdbool.h>
#include <stdio.h>

int main() {
    // Initialise structure
    FusionAhrs ahrs;
    FusionAhrsInitialise(&ahrs);

    FusionAhrsSettings settings = fusionAhrsDefaultSettings;
    settings.sampleRate = 100.0f; // Hz

    FusionAhrsSetSettings(&ahrs, &settings);

    // This loop should repeat for each new gyroscope measurement
    while (true) {
        // Read sensors (replace with actual sensor data)
        const FusionVector gyroscope = {0.0f, 0.0f, 0.0f};
        const FusionVector accelerometer = {0.0f, 0.0f, 1.0f};

        // Update AHRS algorithm
        FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer);

        // Print Euler angles
        const FusionEuler euler = FusionQuaternionToEuler(FusionAhrsGetQuaternion(&ahrs));

        printf("Roll %0.1f, Pitch %0.1f, Yaw %0.1f\n", euler.angle.roll, euler.angle.pitch, euler.angle.yaw);
    }
}
