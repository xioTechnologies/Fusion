/**
 * @file FusionOffset.c
 * @author Seb Madgwick
 * @brief Run-time estimation and compensation of gyroscope offset.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionOffset.h"
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
 * @brief Initialises the offset structure.
 * @param offset Offset structure.
 * @param sampleRate Sample rate in Hz.
 */
void FusionOffsetInitialise(FusionOffset *const offset, const unsigned int sampleRate) {
    offset->filterCoefficient = 2.0f * (float) M_PI * CUTOFF_FREQUENCY * (1.0f / (float) sampleRate);
    offset->timeout = TIMEOUT * sampleRate;
    offset->timer = 0;
    offset->gyroscopeOffset = FUSION_VECTOR_ZERO;
}

/**
 * @brief Updates the offset algorithm and returns the offset-corrected
 * gyroscope.
 * @param offset Offset structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @return Offset-corrected gyroscope in degrees per second.
 */
FusionVector FusionOffsetUpdate(FusionOffset *const offset, FusionVector gyroscope) {
    // Apply gyroscope offset
    gyroscope = FusionVectorSubtract(gyroscope, offset->gyroscopeOffset);

    // Reset timer if gyroscope not stationary
    if ((fabsf(gyroscope.axis.x) > THRESHOLD) || (fabsf(gyroscope.axis.y) > THRESHOLD) || (fabsf(gyroscope.axis.z) > THRESHOLD)) {
        offset->timer = 0;
        return gyroscope;
    }

    // Increment timer while gyroscope stationary
    if (offset->timer < offset->timeout) {
        offset->timer++;
        return gyroscope;
    }

    // Update high-pass filter while timer has elapsed
    offset->gyroscopeOffset = FusionVectorAdd(offset->gyroscopeOffset, FusionVectorScale(gyroscope, offset->filterCoefficient));
    return gyroscope;
}

//------------------------------------------------------------------------------
// End of file
