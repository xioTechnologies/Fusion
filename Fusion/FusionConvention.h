/**
 * @file FusionConvention.h
 * @author Seb Madgwick
 * @brief Earth axes convention.
 */

#ifndef FUSION_CONVENTION_H
#define FUSION_CONVENTION_H

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Earth axes convention describing the direction of the earth X, Y, and
 * Z axes. For example, NWU means that X is pointing north, Y west, and Z up.
 */
typedef enum {
    FusionConventionNwu, /* North, West, Up (NWU) */
    FusionConventionEnu, /* East, North, Up (ENU) */
    FusionConventionNed, /* North, East, Down (NED) */
} FusionConvention;

//------------------------------------------------------------------------------
// Function declarations

const char *FusionConventionToString(const FusionConvention convention);

#endif

//------------------------------------------------------------------------------
// End of file
