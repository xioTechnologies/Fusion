/**
 * @file FusionCompass.c
 * @author Seb Madgwick
 * @brief Tilt-compensated compass to calculate magnetic heading.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionCompass.h"
#include <math.h>

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Calculates magnetic heading.
 * @param convention Earth axes convention.
 * @param accelerometer Accelerometer in any calibrated units.
 * @param magnetometer Magnetometer in any calibrated units.
 * @return Heading in degrees.
 */
float FusionCompass(const FusionConvention convention, const FusionVector accelerometer, const FusionVector magnetometer) {
    switch (convention) {
        case FusionConventionNwu: {
            const FusionVector west = FusionVectorNormalise(FusionVectorCrossProduct(accelerometer, magnetometer));
            const FusionVector north = FusionVectorNormalise(FusionVectorCrossProduct(west, accelerometer));
            return FusionRadiansToDegrees(atan2f(west.axis.x, north.axis.x));
        }
        case FusionConventionEnu: {
            const FusionVector west = FusionVectorNormalise(FusionVectorCrossProduct(accelerometer, magnetometer));
            const FusionVector north = FusionVectorNormalise(FusionVectorCrossProduct(west, accelerometer));
            const FusionVector east = FusionVectorMultiplyScalar(west, -1.0f);
            return FusionRadiansToDegrees(atan2f(north.axis.x, east.axis.x));
        }
        case FusionConventionNed: {
            const FusionVector up = FusionVectorMultiplyScalar(accelerometer, -1.0f);
            const FusionVector west = FusionVectorNormalise(FusionVectorCrossProduct(up, magnetometer));
            const FusionVector north = FusionVectorNormalise(FusionVectorCrossProduct(west, up));
            return FusionRadiansToDegrees(atan2f(west.axis.x, north.axis.x));
        }
    }
    return 0.0f; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
