#include "Fusion.h"
#include <stdbool.h>
#include <stdio.h>

#define SAMPLE_PERIOD (0.01f) // replace this with actual sample period

int main() {
    FusionAhrs ahrs;
    FusionAhrsInitialise(&ahrs);

    // this loop should repeat each time new gyroscope data is available
    while (true) {
        const FusionVector gyroscope = {0.0f, 0.0f, 0.0f}; // replace this with actual gyroscope data in degrees/s
        const FusionVector accelerometer = {0.0f, 0.0f, 1.0f}; // replace this with actual accelerometer data in g

        FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, SAMPLE_PERIOD);

        const FusionEuler euler = FusionQuaternionToEuler(FusionAhrsGetQuaternion(&ahrs));

        printf("Roll %0.1f, Pitch %0.1f, Yaw %0.1f\n", euler.angle.roll, euler.angle.pitch, euler.angle.yaw);
    }
}
