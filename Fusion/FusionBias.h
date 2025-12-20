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
 * @brief Settings.
 */
typedef struct {
    float sampleRate; // Hz
    float stationaryThreshold; // degrees per second
    float stationaryPeriod; // seconds
} FusionBiasSettings;

/**
 * @brief Bias structure. All members are private.
 */
typedef struct {
    FusionBiasSettings settings;
    float filterCoefficient;
    unsigned int timeout;
    unsigned int timer;
    FusionVector offset;
} FusionBias;

//------------------------------------------------------------------------------
// Variable declarations

extern const FusionBiasSettings fusionBiasDefaultSettings;

//------------------------------------------------------------------------------
// Function declarations

void FusionBiasInitialise(FusionBias *const bias);

void FusionBiasSetSettings(FusionBias *const bias, const FusionBiasSettings *const settings);

FusionVector FusionBiasUpdate(FusionBias *const bias, FusionVector gyroscope);

FusionVector FusionBiasGetOffset(const FusionBias *const bias);

void FusionBiasSetOffset(FusionBias *const bias, const FusionVector offset);

#endif

//------------------------------------------------------------------------------
// End of file
