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

/**
 * @brief Timeout in seconds.
 */
#define TIMEOUT (5)

/**
 * @brief Threshold in degrees per second.
 */
#define THRESHOLD (3.0f)

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the bias structure.
 * @param bias Bias structure.
 * @param sampleRate Sample rate in Hz.
 */
void FusionBiasInitialise(FusionBias *const bias, const unsigned int sampleRate) {
    bias->filterCoefficient = 2.0f * (float) M_PI * CUTOFF_FREQUENCY * (1.0f / (float) sampleRate);
    bias->timeout = TIMEOUT * sampleRate;
    bias->timer = 0;
    bias->gyroscopeOffset = FUSION_VECTOR_ZERO;
}

/**
 * @brief Updates the bias algorithm and returns the offset-corrected
 * gyroscope.
 * @param bias Bias structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @return Offset-corrected gyroscope in degrees per second.
 */
FusionVector FusionBiasUpdate(FusionBias *const bias, FusionVector gyroscope) {
    // Apply gyroscope offset
    gyroscope = FusionVectorSubtract(gyroscope, bias->gyroscopeOffset);

    // Reset timer if gyroscope not stationary
    if ((fabsf(gyroscope.axis.x) > THRESHOLD) || (fabsf(gyroscope.axis.y) > THRESHOLD) || (fabsf(gyroscope.axis.z) > THRESHOLD)) {
        bias->timer = 0;
        return gyroscope;
    }

    // Increment timer while gyroscope stationary
    if (bias->timer < bias->timeout) {
        bias->timer++;
        return gyroscope;
    }

    // Update high-pass filter while timer has elapsed
    bias->gyroscopeOffset = FusionVectorAdd(bias->gyroscopeOffset, FusionVectorScale(gyroscope, bias->filterCoefficient));
    return gyroscope;
}

//------------------------------------------------------------------------------
// End of file
