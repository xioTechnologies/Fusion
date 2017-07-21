/**
 * @file FusionAhrs.c
 * @author Seb Madgwick
 * @brief The AHRS sensor fusion algorithm to combines gyroscope, accelerometer,
 * and magnetometer measurements into a single measurement of orientation
 * relative to the Earth (NWU convention).
 * 
 * The algorithm can be used with only gyroscope and accelerometer measurements,
 * or only gyroscope measurements.  Measurements of orientation obtained without
 * magnetometer measurements can be expected to drift in the yaw component of
 * orientation only.  Measurements of orientation obtained without magnetometer
 * and accelerometer measurements can be expected to drift in all three degrees
 * of freedom.
 * 
 * The algorithm also provides a measurement of linear acceleration and Earth
 * acceleration.  Linear acceleration is equal to the accelerometer  measurement
 * with the 1 g of gravity subtracted.  Earth acceleration is a measurement of
 * linear acceleration in the Earth coordinate frame.
 * 
 * The algorithm outputs a quaternion describing the Earth relative to the
 * sensor.  The library includes a quaternion conjugate function for converting
 * this to a quaternion describing the sensor relative to the Earth, as well as
 * functions for converting a quaternion to a rotation matrix or Euler angles.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionAhrs.h"
#include <math.h> // atan2f, cosf, sinf

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Initial gain used during the initialisation period.  The gain used by
 * each algorithm iteration will ramp down from this initial gain to the
 * specified algorithm gain over the initialisation period.
 */
#define INITIAL_GAIN (10.0f)

/**
 * @brief Initialisation period (in seconds).
 */
#define INITIALISATION_PERIOD (3.0f)

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the AHRS algorithm with application settings.
 *
 * Example use:
 * @code
 * FusionAhrs fusionAhrs;
 * FusionAhrsInitialise(&fusionAhrs, 0.5f, 20.0f, 70.0f); // valid magnetic field defined as 20 uT to 70 uT
 * @endcode
 *
 * @param fusionAhrs Address of the FusionAhrs structure.
 * @param gain Algorithm gain.  Must be equal or greater than zero.  A value of
 * zero will mean that the accelerometer and magnetometer are ignored.
 * Increasing the gain will increase the influence of the accelerometer and
 * magnetometer on the algorithm output.
 * @param minMagneticField Minimum valid magnetic field magnitude in the same
 * units as the magnetometer.  If a measured magnetic field magnitude is above
 * this value then the magnetometer will be ignored for that algorithm update.
 * @param maxMagneticField Maximum valid magnetic field magnitude in the same
 * units as the magnetometer.  If a measured magnetic field magnitude is above
 * this value then the magnetometer will be ignored for that algorithm update.
 */
void FusionAhrsInitialise(FusionAhrs * const fusionAhrs, const float gain, const float minMagneticField, const float maxMagneticField) {
    fusionAhrs->gain = gain;
    fusionAhrs->minMagneticFieldSquared = minMagneticField * minMagneticField;
    fusionAhrs->maxMagneticFieldSquared = maxMagneticField * maxMagneticField;
    fusionAhrs->quaternion = FUSION_QUATERNION_IDENTITY;
    fusionAhrs->linearAcceleration = FUSION_VECTOR3_ZERO;
    fusionAhrs->rampedGain = INITIAL_GAIN;
}

/**
 * @brief Updates the algorithm with the latest sensor measurements.  This
 * function should be called for each new gyroscope measurement where the
 * gyroscope is sampled at the specified sample period.  It is not strictly
 * necessary for accelerometer and magnetometer measurements to be synchronised
 * with gyroscope measurements provided that accelerometer and magnetometer
 * measurements used are the most recent available.
 *
 * Example use:
 * @code
 * const FusionVector3 gyroscope = {
 *    .axis.x = 0.0f,
 *    .axis.y = 0.0f,
 *    .axis.z = 0.0f,
 * }; // literal values should be replaced with sensor measurements
 *
 * const FusionVector3 accelerometer = {
 *    .axis.x = 0.0f,
 *    .axis.y = 0.0f,
 *    .axis.z = 1.0f,
 * }; // literal values should be replaced with sensor measurements
 *
 * const FusionVector3 magnetometer = {
 *    .axis.x = 1.0f,
 *    .axis.y = 0.0f,
 *    .axis.z = 0.0f,
 * }; // literal values should be replaced with sensor measurements
 *
 * FusionAhrsUpdate(&fusionAhrs, gyroscope, accelerometer, magnetometer, 0.01f); // assumes 100 Hz sample rate
 * //FusionAhrsUpdate(&fusionAhrs, gyroscope, accelerometer, FUSION_VECTOR3_ZERO, 0.01f); // alternative function call to ignore magnetometer
 * //FusionAhrsUpdate(&fusionAhrs, gyroscope, FUSION_VECTOR3_ZERO, FUSION_VECTOR3_ZERO, 0.01f); // alternative function call to ignore accelerometer and magnetometer
 * @endcode
 *
 * @param fusionAhrs Address of the FusionAhrs structure.
 * @param gyroscope Gyroscope measurement in degrees per second.
 * @param accelerometer Accelerometer measurement in g.
 * @param magnetometer Magnetometer measurement in uT.
 * @param samplePeriod Sample period in seconds.
 */
void FusionAhrsUpdate(FusionAhrs * const fusionAhrs, const FusionVector3 gyroscope, const FusionVector3 accelerometer, const FusionVector3 magnetometer, const float samplePeriod) {
#define Q fusionAhrs->quaternion.element // define shorthand label for more readable code

    // Calculate feedback error
    FusionVector3 halfFeedbackError = FUSION_VECTOR3_ZERO; // scaled by 0.5 to avoid repeated multiplications by 2
    do {
        // Abandon feedback calculation if accelerometer measurement invalid
        if ((accelerometer.axis.x == 0.0f) && (accelerometer.axis.y == 0.0f) && (accelerometer.axis.z == 0.0f)) {
            break;
        }

        // Calculate direction of gravity assumed by quaternion
        const FusionVector3 halfGravity = {
            .axis.x = Q.x * Q.z - Q.w * Q.y,
            .axis.y = Q.w * Q.x + Q.y * Q.z,
            .axis.z = Q.w * Q.w - 0.5f + Q.z * Q.z,
        }; // equal to 3rd column of rotation matrix representation scaled by 0.5

        // Calculate accelerometer feedback error
        halfFeedbackError = FusionVectorCrossProduct(FusionVectorFastNormalise(accelerometer), halfGravity);

        // Abandon magnetometer feedback calculation if magnetometer measurement invalid
        const float magnetometerNorm = magnetometer.axis.x * magnetometer.axis.x
                + magnetometer.axis.y * magnetometer.axis.y
                + magnetometer.axis.z * magnetometer.axis.z;
        if ((magnetometerNorm < fusionAhrs->minMagneticFieldSquared) || (magnetometerNorm > fusionAhrs->maxMagneticFieldSquared)) {
            break;
        }

        // Compute direction of 'magnetic west' assumed by quaternion
        const FusionVector3 halfEast = {
            .axis.x = Q.x * Q.y + Q.w * Q.z,
            .axis.y = Q.w * Q.w - 0.5f + Q.y * Q.y,
            .axis.z = Q.y * Q.z - Q.w * Q.x
        }; // equal to 2nd column of rotation matrix representation scaled by 0.5

        // Calculate magnetometer feedback error
        halfFeedbackError = FusionVectorAdd(halfFeedbackError, FusionVectorCrossProduct(FusionVectorFastNormalise(FusionVectorCrossProduct(accelerometer, magnetometer)), halfEast));

    } while (false);

    // Ramp down gain until initialisation complete
    if (fusionAhrs->gain == 0) {
        fusionAhrs->rampedGain = 0; // skip initialisation if gain is zero
    }
    float feedbackGain = fusionAhrs->gain;
    if (fusionAhrs->rampedGain > fusionAhrs->gain) {
        fusionAhrs->rampedGain -= (INITIAL_GAIN - fusionAhrs->gain) * samplePeriod / INITIALISATION_PERIOD;
        feedbackGain = fusionAhrs->rampedGain;
    }

    // Convert gyroscope to radians per second scaled by 0.5
    FusionVector3 halfGyroscope = FusionVectorMultiplyScalar(gyroscope, 0.5f * FUSION_DEGREES_TO_RADIANS(1));

    // Apply feedback to gyroscope
    halfGyroscope = FusionVectorAdd(halfGyroscope, FusionVectorMultiplyScalar(halfFeedbackError, feedbackGain));

    // Integrate rate of change of quaternion
    fusionAhrs->quaternion = FusionQuaternionAdd(fusionAhrs->quaternion, FusionQuaternionMultiplyVector(fusionAhrs->quaternion, FusionVectorMultiplyScalar(halfGyroscope, samplePeriod)));

    // Normalise quaternion
    fusionAhrs->quaternion = FusionQuaternionFastNormalise(fusionAhrs->quaternion);

    // Calculate linear acceleration
    const FusionVector3 gravity = {
        .axis.x = 2.0f * (Q.x * Q.z - Q.w * Q.y),
        .axis.y = 2.0f * (Q.w * Q.x + Q.y * Q.z),
        .axis.z = 2.0f * (Q.w * Q.w - 0.5f + Q.z * Q.z),
    }; // equal to 3rd column of rotation matrix representation
    fusionAhrs->linearAcceleration = FusionVectorSubtract(accelerometer, gravity);

#undef Q // undefine shorthand label
}

/**
 * @brief Calculates the Earth acceleration for the most recent update of the
 * AHRS algorithm.
 *
 * Example use:
 * @code
 * const FusionVector3 earthAcceleration = FusionAhrsCalculateEarthAcceleration(&fusionAhrs);
 * @endcode
 *
 * @param fusionAhrs Address of the FusionAhrs structure.
 * @return Earth acceleration in g.
 */
FusionVector3 FusionAhrsCalculateEarthAcceleration(const FusionAhrs * const fusionAhrs) {
#define Q fusionAhrs->quaternion.element // define shorthand labels for more readable code
#define A fusionAhrs->linearAcceleration.axis
    const float qwqw = Q.w * Q.w; // calculate common terms to avoid repeated operations
    const float qwqx = Q.w * Q.x;
    const float qwqy = Q.w * Q.y;
    const float qwqz = Q.w * Q.z;
    const float qxqy = Q.x * Q.y;
    const float qxqz = Q.x * Q.z;
    const float qyqz = Q.y * Q.z;
    const FusionVector3 earthAcceleration = {
        .axis.x = 2.0f * ((qwqw - 0.5f + Q.x * Q.x) * A.x + (qxqy - qwqz) * A.y + (qxqz + qwqy) * A.z),
        .axis.y = 2.0f * ((qxqy + qwqz) * A.x + (qwqw - 0.5f + Q.y * Q.y) * A.y + (qyqz - qwqx) * A.z),
        .axis.z = 2.0f * ((qxqz - qwqy) * A.x + (qyqz + qwqx) * A.y + (qwqw - 0.5f + Q.z * Q.z) * A.z),
    }; // transpose of a rotation matrix representation of the quaternion multiplied with the linear acceleration
    return earthAcceleration;
#undef Q // undefine shorthand label
#undef A
}

/**
 * @brief Returns true while the algorithm is initialising.
 *
 * Example use:
 * @code
 * if (FusionAhrsIsInitialising(&fusionAhrs) == true) {
 *     // AHRS algorithm is initialising
 * } else {
 *     // AHRS algorithm is not initialising
 * }
 * @endcode
 *
 * @param fusionAhrs Address of the FusionAhrs structure.
 * @return True while the algorithm is initialising.
 */
bool FusionAhrsIsInitialising(const FusionAhrs * const fusionAhrs) {
    return fusionAhrs->rampedGain > fusionAhrs->gain;
}

/**
 * @brief Reinitialises the AHRS algorithm.
 *
 * Example use:
 * @code
 * FusionAhrsReinitialise(&fusionAhrs);
 * @endcode
 *
 * @param fusionAhrs Address of the FusionAhrs structure.
 */
void FusionAhrsReinitialise(FusionAhrs * const fusionAhrs) {
    fusionAhrs->quaternion = FUSION_QUATERNION_IDENTITY;
    fusionAhrs->linearAcceleration = FUSION_VECTOR3_ZERO;
    fusionAhrs->rampedGain = INITIAL_GAIN;
}

/**
 * @brief Zeros the yaw component of orientation only.  This function should
 * only be used if the AHRS algorithm is being used without a magnetometer.
 *
 * Example use:
 * @code
 * FusionAhrsZeroYaw(&fusionAhrs);
 * @endcode
 *
 * @param fusionAhrs Address of the FusionAhrs structure.
 */
void FusionAhrsZeroYaw(FusionAhrs * const fusionAhrs) {
#define Q fusionAhrs->quaternion.element // define shorthand label for more readable code
    fusionAhrs->quaternion = FusionQuaternionNormalise(fusionAhrs->quaternion); // quaternion must be normalised accurately (approximation not sufficient)
    const float halfInverseYaw = 0.5f * atan2f(Q.x * Q.y + Q.w * Q.z, Q.w * Q.w - 0.5f + Q.x * Q.x); // Euler angle of conjugate
    const FusionQuaternion inverseYawQuaternion = {
        .element.w = cosf(halfInverseYaw),
        .element.x = 0.0f,
        .element.y = 0.0f,
        .element.z = -1.0f * sinf(halfInverseYaw),
    };
    fusionAhrs->quaternion = FusionQuaternionMultiply(inverseYawQuaternion, fusionAhrs->quaternion);
#undef Q // undefine shorthand label
}

//------------------------------------------------------------------------------
// End of file
