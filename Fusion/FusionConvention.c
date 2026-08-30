/**
 * @file FusionConvention.c
 * @author Seb Madgwick
 * @brief Earth axes convention.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionConvention.h"

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Returns a string representation of the earth axes convention.
 * @param convention Earth axes convention.
 * @return String representation of the earth axes convention.
 */
const char *FusionConventionToString(const FusionConvention convention) {
    switch (convention) {
        case FusionConventionNwu:
            return "North, West, Up (NWU)";
        case FusionConventionEnu:
            return "East, North, Up (ENU)";
        case FusionConventionNed:
            return "North, East, Down (NED)";
    }
    return ""; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
