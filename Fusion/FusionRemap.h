/**
 * @file FusionRemap.h
 * @author Seb Madgwick, minor edits by Sam Procter
 * @brief Remaps the sensor axes to the body frame.
 */

#ifndef FUSION_REMAP_H
#define FUSION_REMAP_H

//------------------------------------------------------------------------------
// Includes

#include "FusionInline.h"
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
	FusionRemapAlignmentPXPYPZ, /* +X+Y+Z (remap disabled) */
	FusionRemapAlignmentPXPYNZ, /* +X+Y-Z */
	FusionRemapAlignmentPXPZPY, /* +X+Z+Y */
	FusionRemapAlignmentPXPZNY, /* +X+Z-Y */
	FusionRemapAlignmentPXNYPZ, /* +X-Y+Z */
	FusionRemapAlignmentPXNYNZ, /* +X-Y-Z */
	FusionRemapAlignmentPXNZPY, /* +X-Z+Y */
	FusionRemapAlignmentPXNZNY, /* +X-Z-Y */
	FusionRemapAlignmentPYPXNZ, /* +Y+X-Z */
	FusionRemapAlignmentPYPXPZ, /* +Y+X+Z */
	FusionRemapAlignmentPYPZPX, /* +Y+Z+X */
	FusionRemapAlignmentPYPZNX, /* +Y+Z-X */
	FusionRemapAlignmentPYNXNZ, /* +Y-X-Z */
	FusionRemapAlignmentPYNXPZ, /* +Y-X+Z */
	FusionRemapAlignmentPYNZPX, /* +Y-Z+X */
	FusionRemapAlignmentPYNZNX, /* +Y-Z-X */
	FusionRemapAlignmentPZPXNY, /* +Z+X-Y */
	FusionRemapAlignmentPZPXPY, /* +Z+X+Y */
	FusionRemapAlignmentPZPYPX, /* +Z+Y+X */
	FusionRemapAlignmentPZPYNX, /* +Z+Y-X */
	FusionRemapAlignmentPZNXNY, /* +Z-X-Y */
	FusionRemapAlignmentPZNXPY, /* +Z-X+Y */
	FusionRemapAlignmentPZNYNX, /* +Z-Y-X */
	FusionRemapAlignmentPZNYPX, /* +Z-Y+X */
	FusionRemapAlignmentNXPYPZ, /* -X+Y+Z */
	FusionRemapAlignmentNXPYNZ, /* -X+Y-Z */
	FusionRemapAlignmentNXPZPY, /* -X+Z+Y */
	FusionRemapAlignmentNXPZNY, /* -X+Z-Y */
	FusionRemapAlignmentNXNYPZ, /* -X-Y+Z */
	FusionRemapAlignmentNXNYNZ, /* -X-Y-Z */
	FusionRemapAlignmentNXNZPY, /* -X-Z+Y */
	FusionRemapAlignmentNXNZNY, /* -X-Z-Y */
	FusionRemapAlignmentNYPXPZ, /* -Y+X+Z */
	FusionRemapAlignmentNYPXNZ, /* -Y+X-Z */
	FusionRemapAlignmentNYPZPX, /* -Y+Z+X */
	FusionRemapAlignmentNYPZNX, /* -Y+Z-X */
	FusionRemapAlignmentNYNXPZ, /* -Y-X+Z */
	FusionRemapAlignmentNYNXNZ, /* -Y-X-Z */
	FusionRemapAlignmentNYNZPX, /* -Y-Z+X */
	FusionRemapAlignmentNYNZNX, /* -Y-Z-X */
	FusionRemapAlignmentNZPXNY, /* -Z+X-Y */
	FusionRemapAlignmentNZPXPY, /* -Z+X+Y */
	FusionRemapAlignmentNZPYPX, /* -Z+Y+X */
	FusionRemapAlignmentNZPYNX, /* -Z+Y-X */
	FusionRemapAlignmentNZNXNY, /* -Z-X-Y */
	FusionRemapAlignmentNZNXPY, /* -Z-X+Y */
	FusionRemapAlignmentNZNYNX, /* -Z-Y-X */
	FusionRemapAlignmentNZNYPX, /* -Z-Y+X */
} FusionRemapAlignment;

//------------------------------------------------------------------------------
// Function declarations

const char *FusionRemapAlignmentToString(const FusionRemapAlignment alignment);

//------------------------------------------------------------------------------
// Inline functions

/**
 * @brief Remaps the sensor axes to the body frame.
 * @param sensor Sensor.
 * @param alignment Alignment.
 * @return Sensor remapped to the body frame.
 */
static FUSION_INLINE FusionVector FusionRemap(const FusionVector sensor, const FusionRemapAlignment alignment) {
    FusionVector result;
    switch (alignment) {
	case FusionRemapAlignmentPXPYPZ:
		break;
	case FusionRemapAlignmentPXPYNZ:
		result.axis.x = +sensor.axis.x;
		result.axis.y = +sensor.axis.y;
		result.axis.z = -sensor.axis.z;
		return result;
	case FusionRemapAlignmentPXPZPY:
		result.axis.x = +sensor.axis.x;
		result.axis.y = +sensor.axis.z;
		result.axis.z = +sensor.axis.y;
		return result;
	case FusionRemapAlignmentPXPZNY:
		result.axis.x = +sensor.axis.x;
		result.axis.y = +sensor.axis.z;
		result.axis.z = -sensor.axis.y;
		return result;
	case FusionRemapAlignmentPXNYPZ:
		result.axis.x = +sensor.axis.x;
		result.axis.y = -sensor.axis.y;
		result.axis.z = +sensor.axis.z;
		return result;
	case FusionRemapAlignmentPXNYNZ:
		result.axis.x = +sensor.axis.x;
		result.axis.y = -sensor.axis.y;
		result.axis.z = -sensor.axis.z;
		return result;
	case FusionRemapAlignmentPXNZPY:
		result.axis.x = +sensor.axis.x;
		result.axis.y = -sensor.axis.z;
		result.axis.z = +sensor.axis.y;
		return result;
	case FusionRemapAlignmentPXNZNY:
		result.axis.x = +sensor.axis.x;
		result.axis.y = -sensor.axis.z;
		result.axis.z = -sensor.axis.y;
		return result;
	case FusionRemapAlignmentPYPXNZ:
		result.axis.x = +sensor.axis.y;
		result.axis.y = +sensor.axis.x;
		result.axis.z = -sensor.axis.z;
		return result;
	case FusionRemapAlignmentPYPXPZ:
		result.axis.x = +sensor.axis.y;
		result.axis.y = +sensor.axis.x;
		result.axis.z = +sensor.axis.z;
		return result;
	case FusionRemapAlignmentPYPZPX:
		result.axis.x = +sensor.axis.y;
		result.axis.y = +sensor.axis.z;
		result.axis.z = +sensor.axis.x;
		return result;
	case FusionRemapAlignmentPYPZNX:
		result.axis.x = +sensor.axis.y;
		result.axis.y = +sensor.axis.z;
		result.axis.z = -sensor.axis.x;
		return result;
	case FusionRemapAlignmentPYNXNZ:
		result.axis.x = +sensor.axis.y;
		result.axis.y = -sensor.axis.x;
		result.axis.z = -sensor.axis.z;
		return result;
	case FusionRemapAlignmentPYNXPZ:
		result.axis.x = +sensor.axis.y;
		result.axis.y = -sensor.axis.x;
		result.axis.z = +sensor.axis.z;
		return result;
	case FusionRemapAlignmentPYNZPX:
		result.axis.x = +sensor.axis.y;
		result.axis.y = -sensor.axis.z;
		result.axis.z = +sensor.axis.x;
		return result;
	case FusionRemapAlignmentPYNZNX:
		result.axis.x = +sensor.axis.y;
		result.axis.y = -sensor.axis.z;
		result.axis.z = -sensor.axis.x;
		return result;
	case FusionRemapAlignmentPZPXNY:
		result.axis.x = +sensor.axis.z;
		result.axis.y = +sensor.axis.x;
		result.axis.z = -sensor.axis.y;
		return result;
	case FusionRemapAlignmentPZPXPY:
		result.axis.x = +sensor.axis.z;
		result.axis.y = +sensor.axis.x;
		result.axis.z = +sensor.axis.y;
		return result;
	case FusionRemapAlignmentPZPYPX:
		result.axis.x = +sensor.axis.z;
		result.axis.y = +sensor.axis.y;
		result.axis.z = +sensor.axis.x;
		return result;
	case FusionRemapAlignmentPZPYNX:
		result.axis.x = +sensor.axis.z;
		result.axis.y = +sensor.axis.y;
		result.axis.z = -sensor.axis.x;
		return result;
	case FusionRemapAlignmentPZNXNY:
		result.axis.x = +sensor.axis.z;
		result.axis.y = -sensor.axis.x;
		result.axis.z = -sensor.axis.y;
		return result;
	case FusionRemapAlignmentPZNXPY:
		result.axis.x = +sensor.axis.z;
		result.axis.y = -sensor.axis.x;
		result.axis.z = +sensor.axis.y;
		return result;
	case FusionRemapAlignmentPZNYNX:
		result.axis.x = +sensor.axis.z;
		result.axis.y = -sensor.axis.y;
		result.axis.z = -sensor.axis.x;
		return result;
	case FusionRemapAlignmentPZNYPX:
		result.axis.x = +sensor.axis.z;
		result.axis.y = -sensor.axis.y;
		result.axis.z = +sensor.axis.x;
		return result;
	case FusionRemapAlignmentNXPYPZ:
		result.axis.x = -sensor.axis.x;
		result.axis.y = +sensor.axis.y;
		result.axis.z = +sensor.axis.z;
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
	case FusionRemapAlignmentNXPZNY:
		result.axis.x = -sensor.axis.x;
		result.axis.y = +sensor.axis.z;
		result.axis.z = -sensor.axis.y;
		return result;
	case FusionRemapAlignmentNXNYPZ:
		result.axis.x = -sensor.axis.x;
		result.axis.y = -sensor.axis.y;
		result.axis.z = +sensor.axis.z;
		return result;
	case FusionRemapAlignmentNXNYNZ:
		result.axis.x = -sensor.axis.x;
		result.axis.y = -sensor.axis.y;
		result.axis.z = -sensor.axis.z;
		return result;
	case FusionRemapAlignmentNXNZPY:
		result.axis.x = -sensor.axis.x;
		result.axis.y = -sensor.axis.z;
		result.axis.z = +sensor.axis.y;
		return result;
	case FusionRemapAlignmentNXNZNY:
		result.axis.x = -sensor.axis.x;
		result.axis.y = -sensor.axis.z;
		result.axis.z = -sensor.axis.y;
		return result;
	case FusionRemapAlignmentNYPXPZ:
		result.axis.x = -sensor.axis.y;
		result.axis.y = +sensor.axis.x;
		result.axis.z = +sensor.axis.z;
		return result;
	case FusionRemapAlignmentNYPXNZ:
		result.axis.x = -sensor.axis.y;
		result.axis.y = +sensor.axis.x;
		result.axis.z = -sensor.axis.z;
		return result;
	case FusionRemapAlignmentNYPZPX:
		result.axis.x = -sensor.axis.y;
		result.axis.y = +sensor.axis.z;
		result.axis.z = +sensor.axis.x;
		return result;
	case FusionRemapAlignmentNYPZNX:
		result.axis.x = -sensor.axis.y;
		result.axis.y = +sensor.axis.z;
		result.axis.z = -sensor.axis.x;
		return result;
	case FusionRemapAlignmentNYNXPZ:
		result.axis.x = -sensor.axis.y;
		result.axis.y = -sensor.axis.x;
		result.axis.z = +sensor.axis.z;
		return result;
	case FusionRemapAlignmentNYNXNZ:
		result.axis.x = -sensor.axis.y;
		result.axis.y = -sensor.axis.x;
		result.axis.z = -sensor.axis.z;
		return result;
	case FusionRemapAlignmentNYNZPX:
		result.axis.x = -sensor.axis.y;
		result.axis.y = -sensor.axis.z;
		result.axis.z = +sensor.axis.x;
		return result;
	case FusionRemapAlignmentNYNZNX:
		result.axis.x = -sensor.axis.y;
		result.axis.y = -sensor.axis.z;
		result.axis.z = -sensor.axis.x;
		return result;
	case FusionRemapAlignmentNZPXNY:
		result.axis.x = -sensor.axis.z;
		result.axis.y = +sensor.axis.x;
		result.axis.z = -sensor.axis.y;
		return result;
	case FusionRemapAlignmentNZPXPY:
		result.axis.x = -sensor.axis.z;
		result.axis.y = +sensor.axis.x;
		result.axis.z = +sensor.axis.y;
		return result;
	case FusionRemapAlignmentNZPYPX:
		result.axis.x = -sensor.axis.z;
		result.axis.y = +sensor.axis.y;
		result.axis.z = +sensor.axis.x;
		return result;
	case FusionRemapAlignmentNZPYNX:
		result.axis.x = -sensor.axis.z;
		result.axis.y = +sensor.axis.y;
		result.axis.z = -sensor.axis.x;
		return result;
	case FusionRemapAlignmentNZNXNY:
		result.axis.x = -sensor.axis.z;
		result.axis.y = -sensor.axis.x;
		result.axis.z = -sensor.axis.y;
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
	case FusionRemapAlignmentNZNYPX:
		result.axis.x = -sensor.axis.z;
		result.axis.y = -sensor.axis.y;
		result.axis.z = +sensor.axis.x;
		return result;
    }
    return sensor; // avoid compiler warning
}

#endif

//------------------------------------------------------------------------------
// End of file
