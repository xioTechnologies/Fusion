/**
 * @file FusionBias.h
 * @author Seb Madgwick
 * @brief Run-time estimation and compensation of gyroscope offset.
 */

#ifndef FUSION_BIAS_H
#define FUSION_BIAS_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Bias structure. All members are private.
 */
typedef struct {
    float filterCoefficient;
    unsigned int timeout;
    unsigned int timer;
    FusionVector gyroscopeOffset;
} FusionBias;

//------------------------------------------------------------------------------
// Function declarations

void FusionBiasInitialise(FusionBias *const bias, const unsigned int sampleRate);

FusionVector FusionBiasUpdate(FusionBias *const bias, FusionVector gyroscope);

#endif

//------------------------------------------------------------------------------
// End of file
