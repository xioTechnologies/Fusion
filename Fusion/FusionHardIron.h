/**
 * @file FusionHardIron.h
 * @author Seb Madgwick
 * @brief Hard-iron calibration.
 */

#ifndef FUSION_HARD_IRON_H
#define FUSION_HARD_IRON_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"
#include "FusionResult.h"

//------------------------------------------------------------------------------
// Function declarations

FusionResult FusionHardIronSolve(const FusionVector *const samples, const int numberOfSamples, FusionVector *const offset);

#endif

//------------------------------------------------------------------------------
// End of file
