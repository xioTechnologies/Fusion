/**
 * @file FusionCompass.c
 * @author Seb Madgwick
 * @brief Tilt-compensated compass to calculate an heading relative to magnetic
 * north using accelerometer and magnetometer measurements.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionCompass.h"
#include <math.h> // atan2f

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Calculates the heading relative to magnetic north.
 * @param accelerometer Accelerometer measurement in any calibrated units.
 * @param magnetometer Magnetometer measurement in any calibrated units.
 * @return Heading angle in degrees.
 */
float FusionCompassCalculateHeading(const FusionVector accelerometer, const FusionVector magnetometer) {

    // Compute direction of magnetic west (Earth's y axis)
    const FusionVector magneticWest = FusionVectorNormalise(FusionVectorCrossProduct(accelerometer, magnetometer));

    // Compute direction of magnetic north (Earth's x axis)
    const FusionVector magneticNorth = FusionVectorNormalise(FusionVectorCrossProduct(magneticWest, accelerometer));

    // Calculate angular heading relative to magnetic north
    return FusionRadiansToDegrees(atan2f(magneticWest.axis.x, magneticNorth.axis.x));
}

//------------------------------------------------------------------------------
// End of file
