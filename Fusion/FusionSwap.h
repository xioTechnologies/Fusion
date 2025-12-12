/**
 * @file FusionSwap.h
 * @author Seb Madgwick
 * @brief Swaps the sensor axes for an alternative alignment.
 */

#ifndef FUSION_SWAP_H
#define FUSION_SWAP_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Axes alignment describing the sensor measurement axes relative to the
 * body axes. For example, if the body X axis is aligned with the sensor Y axis
 * and the body Y axis is aligned with sensor X axis but pointing the opposite
 * direction then alignment is +Y-X+Z.
 */
typedef enum {
    FusionSwapAlignmentPXPYPZ, /* +X+Y+Z */
    FusionSwapAlignmentPXNZPY, /* +X-Z+Y */
    FusionSwapAlignmentPXNYNZ, /* +X-Y-Z */
    FusionSwapAlignmentPXPZNY, /* +X+Z-Y */
    FusionSwapAlignmentNXPYNZ, /* -X+Y-Z */
    FusionSwapAlignmentNXPZPY, /* -X+Z+Y */
    FusionSwapAlignmentNXNYPZ, /* -X-Y+Z */
    FusionSwapAlignmentNXNZNY, /* -X-Z-Y */
    FusionSwapAlignmentPYNXPZ, /* +Y-X+Z */
    FusionSwapAlignmentPYNZNX, /* +Y-Z-X */
    FusionSwapAlignmentPYPXNZ, /* +Y+X-Z */
    FusionSwapAlignmentPYPZPX, /* +Y+Z+X */
    FusionSwapAlignmentNYPXPZ, /* -Y+X+Z */
    FusionSwapAlignmentNYNZPX, /* -Y-Z+X */
    FusionSwapAlignmentNYNXNZ, /* -Y-X-Z */
    FusionSwapAlignmentNYPZNX, /* -Y+Z-X */
    FusionSwapAlignmentPZPYNX, /* +Z+Y-X */
    FusionSwapAlignmentPZPXPY, /* +Z+X+Y */
    FusionSwapAlignmentPZNYPX, /* +Z-Y+X */
    FusionSwapAlignmentPZNXNY, /* +Z-X-Y */
    FusionSwapAlignmentNZPYPX, /* -Z+Y+X */
    FusionSwapAlignmentNZNXPY, /* -Z-X+Y */
    FusionSwapAlignmentNZNYNX, /* -Z-Y-X */
    FusionSwapAlignmentNZPXNY, /* -Z+X-Y */
} FusionSwapAlignment;

//------------------------------------------------------------------------------
// Inline functions

/**
 * @brief Swaps the sensor axes for alignment with the body axes.
 * @param sensor Sensor measurement.
 * @param alignment Alignment of sensor relative to body.
 * @return Sensor measurement aligned with the body axes.
 */
static inline FusionVector FusionSwap(const FusionVector sensor, const FusionSwapAlignment alignment) {
    FusionVector result;
    switch (alignment) {
        case FusionSwapAlignmentPXPYPZ:
            return sensor;
        case FusionSwapAlignmentPXNZPY:
            result.axis.x = +sensor.axis.x;
            result.axis.y = -sensor.axis.z;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionSwapAlignmentPXNYNZ:
            result.axis.x = +sensor.axis.x;
            result.axis.y = -sensor.axis.y;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionSwapAlignmentPXPZNY:
            result.axis.x = +sensor.axis.x;
            result.axis.y = +sensor.axis.z;
            result.axis.z = -sensor.axis.y;
            return result;
        case FusionSwapAlignmentNXPYNZ:
            result.axis.x = -sensor.axis.x;
            result.axis.y = +sensor.axis.y;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionSwapAlignmentNXPZPY:
            result.axis.x = -sensor.axis.x;
            result.axis.y = +sensor.axis.z;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionSwapAlignmentNXNYPZ:
            result.axis.x = -sensor.axis.x;
            result.axis.y = -sensor.axis.y;
            result.axis.z = +sensor.axis.z;
            return result;
        case FusionSwapAlignmentNXNZNY:
            result.axis.x = -sensor.axis.x;
            result.axis.y = -sensor.axis.z;
            result.axis.z = -sensor.axis.y;
            return result;
        case FusionSwapAlignmentPYNXPZ:
            result.axis.x = +sensor.axis.y;
            result.axis.y = -sensor.axis.x;
            result.axis.z = +sensor.axis.z;
            return result;
        case FusionSwapAlignmentPYNZNX:
            result.axis.x = +sensor.axis.y;
            result.axis.y = -sensor.axis.z;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionSwapAlignmentPYPXNZ:
            result.axis.x = +sensor.axis.y;
            result.axis.y = +sensor.axis.x;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionSwapAlignmentPYPZPX:
            result.axis.x = +sensor.axis.y;
            result.axis.y = +sensor.axis.z;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionSwapAlignmentNYPXPZ:
            result.axis.x = -sensor.axis.y;
            result.axis.y = +sensor.axis.x;
            result.axis.z = +sensor.axis.z;
            return result;
        case FusionSwapAlignmentNYNZPX:
            result.axis.x = -sensor.axis.y;
            result.axis.y = -sensor.axis.z;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionSwapAlignmentNYNXNZ:
            result.axis.x = -sensor.axis.y;
            result.axis.y = -sensor.axis.x;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionSwapAlignmentNYPZNX:
            result.axis.x = -sensor.axis.y;
            result.axis.y = +sensor.axis.z;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionSwapAlignmentPZPYNX:
            result.axis.x = +sensor.axis.z;
            result.axis.y = +sensor.axis.y;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionSwapAlignmentPZPXPY:
            result.axis.x = +sensor.axis.z;
            result.axis.y = +sensor.axis.x;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionSwapAlignmentPZNYPX:
            result.axis.x = +sensor.axis.z;
            result.axis.y = -sensor.axis.y;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionSwapAlignmentPZNXNY:
            result.axis.x = +sensor.axis.z;
            result.axis.y = -sensor.axis.x;
            result.axis.z = -sensor.axis.y;
            return result;
        case FusionSwapAlignmentNZPYPX:
            result.axis.x = -sensor.axis.z;
            result.axis.y = +sensor.axis.y;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionSwapAlignmentNZNXPY:
            result.axis.x = -sensor.axis.z;
            result.axis.y = -sensor.axis.x;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionSwapAlignmentNZNYNX:
            result.axis.x = -sensor.axis.z;
            result.axis.y = -sensor.axis.y;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionSwapAlignmentNZPXNY:
            result.axis.x = -sensor.axis.z;
            result.axis.y = +sensor.axis.x;
            result.axis.z = -sensor.axis.y;
            return result;
    }
    return sensor; // avoid compiler warning
}

#endif

//------------------------------------------------------------------------------
// End of file
