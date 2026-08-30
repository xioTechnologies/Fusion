/**
 * @file FusionRemap.c
 * @author Seb Madgwick
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
        case FusionRemapAlignmentPXPZNY:
            return "+X+Z-Y";
        case FusionRemapAlignmentPXNZPY:
            return "+X-Z+Y";
        case FusionRemapAlignmentPXNYNZ:
            return "+X-Y-Z";
        case FusionRemapAlignmentPYPXNZ:
            return "+Y+X-Z";
        case FusionRemapAlignmentPYPZPX:
            return "+Y+Z+X";
        case FusionRemapAlignmentPYNZNX:
            return "+Y-Z-X";
        case FusionRemapAlignmentPYNXPZ:
            return "+Y-X+Z";
        case FusionRemapAlignmentPZPXPY:
            return "+Z+X+Y";
        case FusionRemapAlignmentPZPYNX:
            return "+Z+Y-X";
        case FusionRemapAlignmentPZNYPX:
            return "+Z-Y+X";
        case FusionRemapAlignmentPZNXNY:
            return "+Z-X-Y";
        case FusionRemapAlignmentNZPXNY:
            return "-Z+X-Y";
        case FusionRemapAlignmentNZPYPX:
            return "-Z+Y+X";
        case FusionRemapAlignmentNZNYNX:
            return "-Z-Y-X";
        case FusionRemapAlignmentNZNXPY:
            return "-Z-X+Y";
        case FusionRemapAlignmentNYPXPZ:
            return "-Y+X+Z";
        case FusionRemapAlignmentNYPZNX:
            return "-Y+Z-X";
        case FusionRemapAlignmentNYNZPX:
            return "-Y-Z+X";
        case FusionRemapAlignmentNYNXNZ:
            return "-Y-X-Z";
        case FusionRemapAlignmentNXPYNZ:
            return "-X+Y-Z";
        case FusionRemapAlignmentNXPZPY:
            return "-X+Z+Y";
        case FusionRemapAlignmentNXNZNY:
            return "-X-Z-Y";
        case FusionRemapAlignmentNXNYPZ:
            return "-X-Y+Z";
    }
    return ""; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
