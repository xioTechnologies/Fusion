/**
 * @file FusionRemap.h
 * @author Seb Madgwick
 * @brief Remaps the sensor axes to the body frame.
 */

#ifndef FUSION_REMAP_H
#define FUSION_REMAP_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Alignment of the sensor axes relative to the body frame. For example,
 * if the body X axis is aligned with the sensor Y axis and the body Y axis is
 * aligned with the sensor X axis but pointing the opposite direction, then
 * alignment is +Y-X+Z.
 */
typedef enum {
    FusionRemapAlignmentPXPYPZ, /* +X+Y+Z */
    FusionRemapAlignmentPXNZPY, /* +X-Z+Y */
    FusionRemapAlignmentPXNYNZ, /* +X-Y-Z */
    FusionRemapAlignmentPXPZNY, /* +X+Z-Y */
    FusionRemapAlignmentNXPYNZ, /* -X+Y-Z */
    FusionRemapAlignmentNXPZPY, /* -X+Z+Y */
    FusionRemapAlignmentNXNYPZ, /* -X-Y+Z */
    FusionRemapAlignmentNXNZNY, /* -X-Z-Y */
    FusionRemapAlignmentPYNXPZ, /* +Y-X+Z */
    FusionRemapAlignmentPYNZNX, /* +Y-Z-X */
    FusionRemapAlignmentPYPXNZ, /* +Y+X-Z */
    FusionRemapAlignmentPYPZPX, /* +Y+Z+X */
    FusionRemapAlignmentNYPXPZ, /* -Y+X+Z */
    FusionRemapAlignmentNYNZPX, /* -Y-Z+X */
    FusionRemapAlignmentNYNXNZ, /* -Y-X-Z */
    FusionRemapAlignmentNYPZNX, /* -Y+Z-X */
    FusionRemapAlignmentPZPYNX, /* +Z+Y-X */
    FusionRemapAlignmentPZPXPY, /* +Z+X+Y */
    FusionRemapAlignmentPZNYPX, /* +Z-Y+X */
    FusionRemapAlignmentPZNXNY, /* +Z-X-Y */
    FusionRemapAlignmentNZPYPX, /* -Z+Y+X */
    FusionRemapAlignmentNZNXPY, /* -Z-X+Y */
    FusionRemapAlignmentNZNYNX, /* -Z-Y-X */
    FusionRemapAlignmentNZPXNY, /* -Z+X-Y */
} FusionRemapAlignment;

//------------------------------------------------------------------------------
// Inline functions

/**
 * @brief Remaps the sensor axes to the body frame.
 * @param sensor Sensor.
 * @param alignment Alignment.
 * @return Sensor remapped to the body frame.
 */
static inline FusionVector FusionRemap(const FusionVector sensor, const FusionRemapAlignment alignment) {
    FusionVector result;
    switch (alignment) {
        case FusionRemapAlignmentPXPYPZ:
            break;
        case FusionRemapAlignmentPXNZPY:
            result.axis.x = +sensor.axis.x;
            result.axis.y = -sensor.axis.z;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionRemapAlignmentPXNYNZ:
            result.axis.x = +sensor.axis.x;
            result.axis.y = -sensor.axis.y;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionRemapAlignmentPXPZNY:
            result.axis.x = +sensor.axis.x;
            result.axis.y = +sensor.axis.z;
            result.axis.z = -sensor.axis.y;
            return result;
        case FusionRemapAlignmentNXPYNZ:
            result.axis.x = -sensor.axis.x;
            result.axis.y = +sensor.axis.y;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionRemapAlignmentNXPZPY:
            result.axis.x = -sensor.axis.x;
            result.axis.y = +sensor.axis.z;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionRemapAlignmentNXNYPZ:
            result.axis.x = -sensor.axis.x;
            result.axis.y = -sensor.axis.y;
            result.axis.z = +sensor.axis.z;
            return result;
        case FusionRemapAlignmentNXNZNY:
            result.axis.x = -sensor.axis.x;
            result.axis.y = -sensor.axis.z;
            result.axis.z = -sensor.axis.y;
            return result;
        case FusionRemapAlignmentPYNXPZ:
            result.axis.x = +sensor.axis.y;
            result.axis.y = -sensor.axis.x;
            result.axis.z = +sensor.axis.z;
            return result;
        case FusionRemapAlignmentPYNZNX:
            result.axis.x = +sensor.axis.y;
            result.axis.y = -sensor.axis.z;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionRemapAlignmentPYPXNZ:
            result.axis.x = +sensor.axis.y;
            result.axis.y = +sensor.axis.x;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionRemapAlignmentPYPZPX:
            result.axis.x = +sensor.axis.y;
            result.axis.y = +sensor.axis.z;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionRemapAlignmentNYPXPZ:
            result.axis.x = -sensor.axis.y;
            result.axis.y = +sensor.axis.x;
            result.axis.z = +sensor.axis.z;
            return result;
        case FusionRemapAlignmentNYNZPX:
            result.axis.x = -sensor.axis.y;
            result.axis.y = -sensor.axis.z;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionRemapAlignmentNYNXNZ:
            result.axis.x = -sensor.axis.y;
            result.axis.y = -sensor.axis.x;
            result.axis.z = -sensor.axis.z;
            return result;
        case FusionRemapAlignmentNYPZNX:
            result.axis.x = -sensor.axis.y;
            result.axis.y = +sensor.axis.z;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionRemapAlignmentPZPYNX:
            result.axis.x = +sensor.axis.z;
            result.axis.y = +sensor.axis.y;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionRemapAlignmentPZPXPY:
            result.axis.x = +sensor.axis.z;
            result.axis.y = +sensor.axis.x;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionRemapAlignmentPZNYPX:
            result.axis.x = +sensor.axis.z;
            result.axis.y = -sensor.axis.y;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionRemapAlignmentPZNXNY:
            result.axis.x = +sensor.axis.z;
            result.axis.y = -sensor.axis.x;
            result.axis.z = -sensor.axis.y;
            return result;
        case FusionRemapAlignmentNZPYPX:
            result.axis.x = -sensor.axis.z;
            result.axis.y = +sensor.axis.y;
            result.axis.z = +sensor.axis.x;
            return result;
        case FusionRemapAlignmentNZNXPY:
            result.axis.x = -sensor.axis.z;
            result.axis.y = -sensor.axis.x;
            result.axis.z = +sensor.axis.y;
            return result;
        case FusionRemapAlignmentNZNYNX:
            result.axis.x = -sensor.axis.z;
            result.axis.y = -sensor.axis.y;
            result.axis.z = -sensor.axis.x;
            return result;
        case FusionRemapAlignmentNZPXNY:
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
