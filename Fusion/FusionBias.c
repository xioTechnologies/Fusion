/**
 * @file FusionBias.c
 * @author Seb Madgwick
 * @brief Run-time estimation and compensation of gyroscope offset.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionBias.h"
#include <math.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief High-pass filter cutoff frequency in Hz.
 */
#define CUTOFF_FREQUENCY (0.02f)

//------------------------------------------------------------------------------
// Variables

const FusionBiasSettings fusionBiasDefaultSettings = {
    .sampleRate = 100.0f,
    .stationaryThreshold = 3.0f,
    .stationaryPeriod = 3.0f,
};

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the bias structure.
 * @param bias Bias structure.
 */
void FusionBiasInitialise(FusionBias *const bias) {
    FusionBiasSetSettings(bias, &fusionBiasDefaultSettings);
    bias->timer = 0;
    bias->offset = FUSION_VECTOR_ZERO;
}

/**
 * @brief Sets the settings.
 * @param bias Bias structure.
 * @param settings Settings.
 */
void FusionBiasSetSettings(FusionBias *const bias, const FusionBiasSettings *const settings) {
    bias->settings = *settings;
    bias->filterCoefficient = 2.0f * (float) M_PI * CUTOFF_FREQUENCY * (1.0f / bias->settings.sampleRate);
    bias->timeout = (unsigned int) (bias->settings.stationaryPeriod * bias->settings.sampleRate);
}

/**
 * @brief Updates the bias algorithm and returns the offset-corrected
 * gyroscope. This function must be called for every gyroscope sample at the
 * configured sample rate.
 * @param bias Bias structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @return Offset-corrected gyroscope in degrees per second.
 */
FusionVector FusionBiasUpdate(FusionBias *const bias, FusionVector gyroscope) {
    // Apply gyroscope offset
    gyroscope = FusionVectorSubtract(gyroscope, bias->offset);

    // Reset timer if gyroscope not stationary
    if ((fabsf(gyroscope.axis.x) > bias->settings.stationaryThreshold) ||
        (fabsf(gyroscope.axis.y) > bias->settings.stationaryThreshold) ||
        (fabsf(gyroscope.axis.z) > bias->settings.stationaryThreshold)) {
        bias->timer = 0;
        return gyroscope;
    }

    // Increment timer while gyroscope stationary
    if (bias->timer < bias->timeout) {
        bias->timer++;
        return gyroscope;
    }

    // Update high-pass filter while timer has elapsed
    bias->offset = FusionVectorAdd(bias->offset, FusionVectorScale(gyroscope, bias->filterCoefficient));
    return gyroscope;
}

/**
 * @brief Returns the gyroscope offset.
 * @param bias Bias structure.
 * @return Gyroscope offset in degrees per second.
 */
FusionVector FusionBiasGetOffset(const FusionBias *const bias) {
    return bias->offset;
}

/**
 * @brief Sets the gyroscope offset.
 * @param bias Bias structure.
 * @param offset Gyroscope offset in degrees per second.
 */
void FusionBiasSetOffset(FusionBias *const bias, const FusionVector offset) {
    bias->offset = offset;
}

//------------------------------------------------------------------------------
// End of file
