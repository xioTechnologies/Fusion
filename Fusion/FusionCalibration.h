/**
 * @file FusionCalibration.h
 * @author Seb Madgwick
 * @brief Sensor calibration models.
 */

#ifndef FUSION_CALIBRATION_H
#define FUSION_CALIBRATION_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"

//------------------------------------------------------------------------------
// Inline functions

/**
 * @brief Gyroscope and accelerometer calibration model.
 * @param uncalibrated Uncalibrated gyroscope or accelerometer.
 * @param misalignment Misalignment matrix.
 * @param sensitivity Sensitivity.
 * @param offset Offset.
 * @return Calibrated gyroscope or accelerometer.
 */
static inline FusionVector FusionCalibrationInertial(const FusionVector uncalibrated, const FusionMatrix misalignment, const FusionVector sensitivity, const FusionVector offset) {
    return FusionMatrixMultiplyVector(misalignment, FusionVectorHadamardProduct(FusionVectorSubtract(uncalibrated, offset), sensitivity));
}

/**
 * @brief Magnetometer calibration model.
 * @param uncalibrated Uncalibrated magnetometer.
 * @param softIron Soft-iron matrix.
 * @param hardIron Hard-iron offset.
 * @return Calibrated magnetometer.
 */
static inline FusionVector FusionCalibrationMagnetic(const FusionVector uncalibrated, const FusionMatrix softIron, const FusionVector hardIron) {
    return FusionMatrixMultiplyVector(softIron, FusionVectorSubtract(uncalibrated, hardIron));
}

#endif

//------------------------------------------------------------------------------
// End of file
