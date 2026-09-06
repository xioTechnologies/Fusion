/**
 * @file FusionRemap.c
 * @author Seb Madgwick, minor edits by Sam Procter
 * @brief Remaps the sensor axes to the body frame.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionRemap.h"

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Returns a string representation of the alignment.
 * @param alignment Alignment.
 * @return String representation of the alignment.
 */
const char *FusionRemapAlignmentToString(const FusionRemapAlignment alignment) {
    switch (alignment) {
	case FusionRemapAlignmentPXPYPZ:
		return "+X+Y+Z";
	case FusionRemapAlignmentPXPYNZ:
		return "+X+Y-Z";
	case FusionRemapAlignmentPXPZPY:
		return "+X+Z+Y";
	case FusionRemapAlignmentPXPZNY:
		return "+X+Z-Y";
	case FusionRemapAlignmentPXNYPZ:
		return "+X-Y+Z";
	case FusionRemapAlignmentPXNYNZ:
		return "+X-Y-Z";
	case FusionRemapAlignmentPXNZPY:
		return "+X-Z+Y";
	case FusionRemapAlignmentPXNZNY:
		return "+X-Z-Y";
	case FusionRemapAlignmentPYPXNZ:
		return "+Y+X-Z";
	case FusionRemapAlignmentPYPXPZ:
		return "+Y+X+Z";
	case FusionRemapAlignmentPYPZPX:
		return "+Y+Z+X";
	case FusionRemapAlignmentPYPZNX:
		return "+Y+Z-X";
	case FusionRemapAlignmentPYNXNZ:
		return "+Y-X-Z";
	case FusionRemapAlignmentPYNXPZ:
		return "+Y-X+Z";
	case FusionRemapAlignmentPYNZPX:
		return "+Y-Z+X";
	case FusionRemapAlignmentPYNZNX:
		return "+Y-Z-X";
	case FusionRemapAlignmentPZPXNY:
		return "+Z+X-Y";
	case FusionRemapAlignmentPZPXPY:
		return "+Z+X+Y";
	case FusionRemapAlignmentPZPYPX:
		return "+Z+Y+X";
	case FusionRemapAlignmentPZPYNX:
		return "+Z+Y-X";
	case FusionRemapAlignmentPZNXNY:
		return "+Z-X-Y";
	case FusionRemapAlignmentPZNXPY:
		return "+Z-X+Y";
	case FusionRemapAlignmentPZNYNX:
		return "+Z-Y-X";
	case FusionRemapAlignmentPZNYPX:
		return "+Z-Y+X";
	case FusionRemapAlignmentNXPYPZ:
		return "-X+Y+Z";
	case FusionRemapAlignmentNXPYNZ:
		return "-X+Y-Z";
	case FusionRemapAlignmentNXPZPY:
		return "-X+Z+Y";
	case FusionRemapAlignmentNXPZNY:
		return "-X+Z-Y";
	case FusionRemapAlignmentNXNYPZ:
		return "-X-Y+Z";
	case FusionRemapAlignmentNXNYNZ:
		return "-X-Y-Z";
	case FusionRemapAlignmentNXNZPY:
		return "-X-Z+Y";
	case FusionRemapAlignmentNXNZNY:
		return "-X-Z-Y";
	case FusionRemapAlignmentNYPXPZ:
		return "-Y+X+Z";
	case FusionRemapAlignmentNYPXNZ:
		return "-Y+X-Z";
	case FusionRemapAlignmentNYPZPX:
		return "-Y+Z+X";
	case FusionRemapAlignmentNYPZNX:
		return "-Y+Z-X";
	case FusionRemapAlignmentNYNXPZ:
		return "-Y-X+Z";
	case FusionRemapAlignmentNYNXNZ:
		return "-Y-X-Z";
	case FusionRemapAlignmentNYNZPX:
		return "-Y-Z+X";
	case FusionRemapAlignmentNYNZNX:
		return "-Y-Z-X";
	case FusionRemapAlignmentNZPXNY:
		return "-Z+X-Y";
	case FusionRemapAlignmentNZPXPY:
		return "-Z+X+Y";
	case FusionRemapAlignmentNZPYPX:
		return "-Z+Y+X";
	case FusionRemapAlignmentNZPYNX:
		return "-Z+Y-X";
	case FusionRemapAlignmentNZNXNY:
		return "-Z-X-Y";
	case FusionRemapAlignmentNZNXPY:
		return "-Z-X+Y";
	case FusionRemapAlignmentNZNYNX:
		return "-Z-Y-X";
	case FusionRemapAlignmentNZNYPX:
		return "-Z-Y+X";
    }
    return ""; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
