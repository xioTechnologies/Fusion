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
 * @brief Earth axes convention.
 */
typedef enum {
    FusionConventionNwu, /* North (X), West (Y), Up (Z) */
    FusionConventionEnu, /* East (X), North (Y), Up (Z) */
    FusionConventionNed, /* North (X), East (Y), Down (Z) */
} FusionConvention;

#endif

//------------------------------------------------------------------------------
// End of file
