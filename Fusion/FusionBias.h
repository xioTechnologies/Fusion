/**
 * @file FusionBias.h
 * @author Seb Madgwick
 * @brief Gyroscope bias correction algorithm for run-time calibration of the
 * gyroscope offset.
 */

#ifndef FUSION_BIAS_H
#define FUSION_BIAS_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Gyroscope bias algorithm structure. Structure members are used
 * internally and must not be accessed by the application.
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
