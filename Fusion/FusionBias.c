/**
 * @file FusionBias.c
 * @author Seb Madgwick
 * @brief Run-time estimation and correction of gyroscope offset.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionBias.h"
#include <math.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Filter cutoff frequency in Hz.
 */
#define CUTOFF_FREQUENCY (0.02f)

//------------------------------------------------------------------------------
// Variables

const FusionBiasSettings fusionBiasDefaultSettings = {
    .sampleRate = 100.0f,
    .duration = 15.0f,
    .threshold = 3.0f,
    .continuous = true,
    .holdoff = 3.0f,
};

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the bias structure.
 * @param bias Bias structure.
 */
void FusionBiasInitialise(FusionBias *const bias) {
    FusionBiasSetSettings(bias, &fusionBiasDefaultSettings);

    bias->correctedGyroscope = FUSION_VECTOR_ZERO;
    bias->offset = FUSION_VECTOR_ZERO;
    bias->status = FusionProgressStatusNotStarted;
    bias->completed = false;
    bias->durationTimer = 0;
    bias->holdoffTimer = 0;
}

/**
 * @brief Sets the settings.
 * @param bias Bias structure.
 * @param settings Settings.
 */
void FusionBiasSetSettings(FusionBias *const bias, const FusionBiasSettings *const settings) {
    bias->filterCoefficient = 2.0f * (float) M_PI * CUTOFF_FREQUENCY * (1.0f / settings->sampleRate);
    bias->duration = (uint32_t) (settings->sampleRate * settings->duration);
    bias->threshold = settings->threshold;
    bias->continuous = settings->continuous;
    bias->holdoff = (uint32_t) (settings->sampleRate * settings->holdoff);
}

/**
 * @brief Updates the bias algorithm. This function must be called for every
 * gyroscope measurement at the configured sample rate.
 * @param bias Bias structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @return Result.
 */
FusionResult FusionBiasUpdate(FusionBias *const bias, const FusionVector gyroscope) {
    bias->correctedGyroscope = FusionVectorSubtract(gyroscope, bias->offset);

    if ((fabsf(bias->correctedGyroscope.axis.x) > bias->threshold) ||
        (fabsf(bias->correctedGyroscope.axis.y) > bias->threshold) ||
        (fabsf(bias->correctedGyroscope.axis.z) > bias->threshold)) {
        bias->holdoffTimer = 0;

        if (bias->status == FusionProgressStatusInProgress) {
            bias->status = FusionProgressStatusFailed;
            return FusionResultNotStationary;
        }
        return FusionResultOk;
    }

    bool active = false;

    if (bias->status == FusionProgressStatusInProgress) {
        if (++bias->durationTimer >= bias->duration) {
            FusionBiasComplete(bias);
        } else {
            active = true;
        }
    }

    if (bias->continuous) {
        if (bias->holdoffTimer < bias->holdoff) {
            bias->holdoffTimer++;
        } else {
            active = true;
        }
    }

    if (active) {
        bias->offset = FusionVectorAdd(bias->offset, FusionVectorScale(bias->correctedGyroscope, bias->filterCoefficient));
    }
    return FusionResultOk;
}

/**
 * @brief Returns the corrected gyroscope.
 * @param bias Bias structure.
 * @return Corrected gyroscope in degrees per second.
 */
FusionVector FusionBiasGetCorrectedGyroscope(const FusionBias *const bias) {
    return bias->correctedGyroscope;
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

/**
 * @brief Starts calibration.
 * @param bias Bias structure.
 */
void FusionBiasStart(FusionBias *const bias) {
    bias->status = FusionProgressStatusInProgress;
    bias->completed = false;
    bias->durationTimer = 0;
}

/**
 * @brief Returns the progress.
 * @param bias Bias structure.
 * @return Progress.
 */
FusionProgress FusionBiasGetProgress(const FusionBias *const bias) {
    const unsigned int percentage = (unsigned int) ((100 * bias->durationTimer) / bias->duration);

    const FusionProgress progress = {
        .status = bias->status,
        .percentage = percentage > 100 ? 100 : percentage,
    };
    return progress;
}

/**
 * @brief Completes calibration using the measurements processed so far. This
 * function is not normally called by the application because calibration
 * completes automatically once the duration has elapsed.
 * @param bias Bias structure.
 * @return Result.
 */
FusionResult FusionBiasComplete(FusionBias *const bias) {
    if (bias->status != FusionProgressStatusInProgress) {
        return FusionResultNotInProgress;
    }

    bias->status = FusionProgressStatusComplete;
    bias->completed = true;
    return FusionResultOk;
}

/**
 * @brief Aborts calibration.
 * @param bias Bias structure.
 * @return Result.
 */
FusionResult FusionBiasAbort(FusionBias *const bias) {
    if (bias->status != FusionProgressStatusInProgress) {
        return FusionResultNotInProgress;
    }

    bias->status = FusionProgressStatusAborted;
    return FusionResultOk;
}

/**
 * @brief Returns true if calibration has completed. Calling this function will
 * reset the flag.
 * @param bias Bias structure.
 * @return True if calibration has completed.
 */
bool FusionBiasCompleted(FusionBias *const bias) {
    const bool completed = bias->completed;
    bias->completed = false;
    return completed;
}

//------------------------------------------------------------------------------
// End of file
