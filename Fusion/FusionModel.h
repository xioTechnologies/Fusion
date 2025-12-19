/**
 * @file FusionModel.h
 * @author Seb Madgwick
 * @brief Sensor models for applying calibration parameters.
 */

#ifndef FUSION_MODEL_H
#define FUSION_MODEL_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"

//------------------------------------------------------------------------------
// Inline functions

/**
 * @brief Gyroscope and accelerometer sensor model.
 * @param uncalibrated Uncalibrated gyroscope or accelerometer.
 * @param misalignment Misalignment matrix.
 * @param sensitivity Sensitivity.
 * @param offset Offset.
 * @return Calibrated gyroscope or accelerometer.
 */
static inline FusionVector FusionModelInertial(const FusionVector uncalibrated, const FusionMatrix misalignment, const FusionVector sensitivity, const FusionVector offset) {
    return FusionMatrixMultiply(misalignment, FusionVectorHadamard(FusionVectorSubtract(uncalibrated, offset), sensitivity));
}

/**
 * @brief Magnetometer sensor model.
 * @param uncalibrated Uncalibrated magnetometer.
 * @param softIronMatrix Soft-iron matrix.
 * @param hardIronOffset Hard-iron offset.
 * @return Calibrated magnetometer.
 */
static inline FusionVector FusionModelMagnetic(const FusionVector uncalibrated, const FusionMatrix softIronMatrix, const FusionVector hardIronOffset) {
    return FusionMatrixMultiply(softIronMatrix, FusionVectorSubtract(uncalibrated, hardIronOffset));
}

#endif

//------------------------------------------------------------------------------
// End of file
