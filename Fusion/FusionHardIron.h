/**
 * @file FusionHardIron.h
 * @author Seb Madgwick
 * @brief Run-time estimation and compensation of hard-iron offset.
 */

#ifndef FUSION_HARD_IRON_H
#define FUSION_HARD_IRON_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"
#include "FusionResult.h"

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Settings.
 */
typedef struct {
    float sampleRate; // Hz
    float timeout; // seconds
} FusionHardIronSettings;

/**
 * @brief Hard-iron structure. All members are private.
 */
typedef struct {
    unsigned int timeout;
    FusionVector offset;
} FusionHardIron;

//------------------------------------------------------------------------------
// Variable declarations

extern const FusionHardIronSettings fusionHardIronDefaultSettings;

//------------------------------------------------------------------------------
// Function declarations

void FusionHardIronInitialise(FusionHardIron *const hardIron);

void FusionHardIronSetSettings(FusionHardIron *const hardIron, const FusionHardIronSettings *const settings);

FusionVector FusionHardIronUpdate(FusionHardIron *const hardIron, const FusionVector gyroscope, FusionVector magnetometer);

FusionResult FusionHardIronSolve(const FusionVector *const samples, const int numberOfSamples, FusionVector *const offset);

#endif

//------------------------------------------------------------------------------
// End of file
