#include "Fusion.h"
#include <stdbool.h>
#include <stdio.h>

#define HARD_IRON_TRIGGER() false

int main() {
    const float sampleRate = 100.0f; // Hz

    // Configure AHRS algorithm
    FusionAhrs ahrs;
    FusionAhrsInitialise(&ahrs);

    FusionAhrsSettings settings = fusionAhrsDefaultSettings;
    settings.gyroscopeRange = 2000.0f; // degrees per second
    settings.accelerationRejection = 10.0f; // reject acceleration errors > 10 degrees
    settings.magneticRejection = 10.0f; // reject magnetic errors >10 degrees
    settings.recoveryTriggerPeriod = (unsigned int) (5.0f * sampleRate); // reject acceleration/magnetic disturbances for up to 5 seconds

    FusionAhrsSetSettings(&ahrs, &settings);

    // Configure bias algorithm
    FusionBias bias;
    FusionBiasInitialise(&bias);

    FusionBiasSettings biasSettings = fusionBiasDefaultSettings;
    biasSettings.sampleRate = sampleRate;

    FusionBiasSetSettings(&bias, &biasSettings);

    // Configure hard-iron algorithm
    FusionHardIron hardIron;
    FusionHardIronInitialise(&hardIron);

    FusionHardIronSettings hardIronSettings = fusionHardIronDefaultSettings;
    hardIronSettings.sampleRate = sampleRate;

    FusionHardIronSetSettings(&hardIron, &hardIronSettings);

    // Open CSV file
    FILE *file = fopen(SENSOR_DATA_CSV, "r");

    char header[256];
    fgets(header, sizeof(header), file); // skip CSV header

    while (true) {
        // Read each CSV line as if reading sensor data in real-time
        float seconds;
        FusionVector gyroscope;
        FusionVector accelerometer;
        FusionVector magnetometer;

        if (fscanf(file, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                   &seconds,
                   &gyroscope.axis.x, &gyroscope.axis.y, &gyroscope.axis.z,
                   &accelerometer.axis.x, &accelerometer.axis.y, &accelerometer.axis.z,
                   &magnetometer.axis.x, &magnetometer.axis.y, &magnetometer.axis.z) != 10) {
            break;
        }

        // Calibration parameters
        const FusionMatrix gyroscopeMisalignment = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        };
        const FusionVector gyroscopeSensitivity = {
            1.0f, 1.0f, 1.0f,
        };
        const FusionVector gyroscopeOffset = {
            0.0f, 0.0f, 0.0f,
        };

        const FusionMatrix accelerometerMisalignment = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        };
        const FusionVector accelerometerSensitivity = {
            1.0f, 1.0f, 1.0f,
        };
        const FusionVector accelerometerOffset = {
            0.0f, 0.0f, 0.0f,
        };

        const FusionMatrix softIronMatrix = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        };
        const FusionVector hardIronOffset = {
            0.0f, 0.0f, 0.0f,
        };

        // Apply calibration
        gyroscope = FusionModelInertial(gyroscope, gyroscopeMisalignment, gyroscopeSensitivity, gyroscopeOffset);

        accelerometer = FusionModelInertial(accelerometer, accelerometerMisalignment, accelerometerSensitivity, accelerometerOffset);

        magnetometer = FusionModelMagnetic(magnetometer, softIronMatrix, hardIronOffset);

        // Update bias algorithm
        gyroscope = FusionBiasUpdate(&bias, gyroscope);

        // Update hard-iron algorithm
        if (HARD_IRON_TRIGGER()) {
            FusionHardIronInitialise(&hardIron);
        }

        magnetometer = FusionHardIronUpdate(&hardIron, gyroscope, magnetometer);

        printf("Hard-iron algorithm status\n");

        // Remap sensor axes
        const FusionRemapAlignment alignment = FusionRemapAlignmentPXPYPZ;

        gyroscope = FusionRemap(gyroscope, alignment);

        accelerometer = FusionRemap(accelerometer, alignment);

        magnetometer = FusionRemap(magnetometer, alignment);

        // Print calibrated and compensated sensors
        printf("Gyroscope (degrees per second) = %0.3f, %0.3f, %0.3f\n",
               gyroscope.axis.x, gyroscope.axis.y, gyroscope.axis.z);

        printf("Accelerometer (g) = %0.3f, %0.3f, %0.3f\n",
               accelerometer.axis.x, accelerometer.axis.y, accelerometer.axis.z);

        printf("Magnetometer (a.u.) = %0.3f, %0.3f, %0.3f\n", /* a.u. = arbitrary units */
               magnetometer.axis.x, magnetometer.axis.y, magnetometer.axis.z);

        // Update AHRS algorithm
        static float previousSeconds;
        FusionAhrsSetSamplePeriod(&ahrs, seconds - previousSeconds);
        previousSeconds = seconds;

        FusionAhrsUpdate(&ahrs, gyroscope, accelerometer, magnetometer);

        // Print AHRS outputs
        const FusionEuler euler = FusionQuaternionToEuler(FusionAhrsGetQuaternion(&ahrs));

        printf("Euler angles (degrees) = %0.1f, %0.1f, %0.1f\n",
               euler.angle.roll, euler.angle.pitch, euler.angle.yaw);

        const FusionVector linearAcceleration = FusionAhrsGetLinearAcceleration(&ahrs);

        printf("Linear acceleration (g) = %0.3f, %0.3f, %0.3f\n",
               linearAcceleration.axis.x, linearAcceleration.axis.y, linearAcceleration.axis.z);

        const FusionVector earthAcceleration = FusionAhrsGetEarthAcceleration(&ahrs);

        printf("Earth acceleration (g) = %0.3f, %0.3f, %0.3f\n",
               earthAcceleration.axis.x, earthAcceleration.axis.y, earthAcceleration.axis.z);

        const FusionVector gravity = FusionAhrsGetGravity(&ahrs);

        printf("Gravity (unit vector) = %0.3f, %0.3f, %0.3f\n",
               gravity.axis.x, gravity.axis.y, gravity.axis.z);
    }

    fclose(file);
    return 0;
}
