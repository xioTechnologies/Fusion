/**
 * @file FusionBias.h
 * @author Seb Madgwick
 * @brief Run-time estimation and correction of gyroscope offset.
 */

#ifndef FUSION_BIAS_H
#define FUSION_BIAS_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"
#include "FusionProgress.h"
#include "FusionResult.h"
#include <stdbool.h>
#include <stdint.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Settings.
 */
typedef struct {
    float sampleRate; // Hz
    float duration; // seconds
    float threshold; // degrees per second
    bool continuous;
    float holdoff; // seconds
} FusionBiasSettings;

/**
 * @brief Bias structure. All members are private.
 */
typedef struct {
    // Settings
    float filterCoefficient;
    uint32_t duration;
    float threshold;
    bool continuous;
    uint32_t holdoff;

    // Outputs
    FusionVector correctedGyroscope;
    FusionVector offset;

    // Internal states
    FusionProgressStatus status;
    bool completed;
    uint32_t durationTimer;
    uint32_t holdoffTimer;
} FusionBias;

//------------------------------------------------------------------------------
// Variable declarations

extern const FusionBiasSettings fusionBiasDefaultSettings;

//------------------------------------------------------------------------------
// Function declarations

void FusionBiasInitialise(FusionBias *const bias);

void FusionBiasSetSettings(FusionBias *const bias, const FusionBiasSettings *const settings);

FusionResult FusionBiasUpdate(FusionBias *const bias, const FusionVector gyroscope);

FusionVector FusionBiasGetCorrectedGyroscope(const FusionBias *const bias);

FusionVector FusionBiasGetOffset(const FusionBias *const bias);

void FusionBiasSetOffset(FusionBias *const bias, const FusionVector offset);

void FusionBiasStart(FusionBias *const bias);

FusionProgress FusionBiasGetProgress(const FusionBias *const bias);

FusionResult FusionBiasComplete(FusionBias *const bias);

FusionResult FusionBiasAbort(FusionBias *const bias);

bool FusionBiasCompleted(FusionBias *const bias);

#endif

//------------------------------------------------------------------------------
// End of file
