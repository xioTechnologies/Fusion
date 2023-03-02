/**
 * @file FusionAhrs.c
 * @author Seb Madgwick
 * @brief AHRS algorithm to combine gyroscope, accelerometer, and magnetometer
 * measurements into a single measurement of orientation relative to the Earth.
 */

//------------------------------------------------------------------------------
// Includes

#include <float.h> // FLT_MAX
#include "FusionAhrs.h"
#include "FusionCompass.h"
#include <math.h> // atan2f, cosf, powf, sinf

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Initial gain used during the initialisation.
 */
#define INITIAL_GAIN (10.0f)

/**
 * @brief Initialisation period in seconds.
 */
#define INITIALISATION_PERIOD (3.0f)

//------------------------------------------------------------------------------
// Function declarations

static FusionVector HalfGravity(const FusionAhrs *const ahrs);

static FusionVector HalfMagnetic(const FusionAhrs *const ahrs);

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the AHRS algorithm structure.
 * @param ahrs AHRS algorithm structure.
 */
void FusionAhrsInitialise(FusionAhrs *const ahrs) {
    const FusionAhrsSettings settings = {
            .convention = FusionConventionNwu,
            .gain = 0.5f,
            .accelerationRejection = 90.0f,
            .magneticRejection = 90.0f,
            .rejectionTimeout = 0,
    };
    FusionAhrsSetSettings(ahrs, &settings);
    FusionAhrsReset(ahrs);
}

/**
 * @brief Resets the AHRS algorithm.  This is equivalent to reinitialising the
 * algorithm while maintaining the current settings.
 * @param ahrs AHRS algorithm structure.
 */
void FusionAhrsReset(FusionAhrs *const ahrs) {
    ahrs->quaternion = FUSION_IDENTITY_QUATERNION;
    ahrs->accelerometer = FUSION_VECTOR_ZERO;
    ahrs->initialising = true;
    ahrs->rampedGain = INITIAL_GAIN;
    ahrs->halfAccelerometerFeedback = FUSION_VECTOR_ZERO;
    ahrs->halfMagnetometerFeedback = FUSION_VECTOR_ZERO;
    ahrs->accelerometerIgnored = false;
    ahrs->accelerationRejectionTimer = 0;
    ahrs->accelerationRejectionTimeout = false;
    ahrs->magnetometerIgnored = false;
    ahrs->magneticRejectionTimer = 0;
    ahrs->magneticRejectionTimeout = false;
}

/**
 * @brief Sets the AHRS algorithm settings.
 * @param ahrs AHRS algorithm structure.
 * @param settings Settings.
 */
void FusionAhrsSetSettings(FusionAhrs *const ahrs, const FusionAhrsSettings *const settings) {
    ahrs->settings.convention = settings->convention;
    ahrs->settings.gain = settings->gain;
    if ((settings->accelerationRejection == 0.0f) || (settings->rejectionTimeout == 0)) {
        ahrs->settings.accelerationRejection = FLT_MAX;
    } else {
        ahrs->settings.accelerationRejection = powf(0.5f * sinf(FusionDegreesToRadians(settings->accelerationRejection)), 2);
    }
    if ((settings->magneticRejection == 0.0f) || (settings->rejectionTimeout == 0)) {
        ahrs->settings.magneticRejection = FLT_MAX;
    } else {
        ahrs->settings.magneticRejection = powf(0.5f * sinf(FusionDegreesToRadians(settings->magneticRejection)), 2);
    }
    ahrs->settings.rejectionTimeout = settings->rejectionTimeout;
    if (ahrs->initialising == false) {
        ahrs->rampedGain = ahrs->settings.gain;
    }
    ahrs->rampedGainStep = (INITIAL_GAIN - ahrs->settings.gain) / INITIALISATION_PERIOD;
}

/**
 * @brief Updates the AHRS algorithm using the gyroscope, accelerometer, and
 * magnetometer measurements.
 * @param ahrs AHRS algorithm structure.
 * @param gyroscope Gyroscope measurement in degrees per second.
 * @param accelerometer Accelerometer measurement in g.
 * @param magnetometer Magnetometer measurement in arbitrary units.
 * @param deltaTime Delta time in seconds.
 */
void FusionAhrsUpdate(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const FusionVector magnetometer, const float deltaTime) {
#define Q ahrs->quaternion.element

    // Store accelerometer
    ahrs->accelerometer = accelerometer;

    // Ramp down gain during initialisation
    if (ahrs->initialising == true) {
        ahrs->rampedGain -= ahrs->rampedGainStep * deltaTime;
        if (ahrs->rampedGain < ahrs->settings.gain) {
            ahrs->rampedGain = ahrs->settings.gain;
            ahrs->initialising = false;
            ahrs->accelerationRejectionTimeout = false;
        }
    }

    // Calculate direction of gravity indicated by algorithm
    const FusionVector halfGravity = HalfGravity(ahrs);

    // Calculate accelerometer feedback
    FusionVector halfAccelerometerFeedback = FUSION_VECTOR_ZERO;
    ahrs->accelerometerIgnored = true;
    if (FusionVectorIsZero(accelerometer) == false) {

        // Enter acceleration recovery state if acceleration rejection times out
        if (ahrs->accelerationRejectionTimer > ahrs->settings.rejectionTimeout) {
            const FusionQuaternion quaternion = ahrs->quaternion;
            FusionAhrsReset(ahrs);
            ahrs->quaternion = quaternion;
            ahrs->accelerationRejectionTimer = 0;
            ahrs->accelerationRejectionTimeout = true;
        }

        // Calculate accelerometer feedback scaled by 0.5
        ahrs->halfAccelerometerFeedback = FusionVectorCrossProduct(FusionVectorNormalise(accelerometer), halfGravity);

        // Ignore accelerometer if acceleration distortion detected
        if ((ahrs->initialising == true) || (FusionVectorMagnitudeSquared(ahrs->halfAccelerometerFeedback) <= ahrs->settings.accelerationRejection)) {
            halfAccelerometerFeedback = ahrs->halfAccelerometerFeedback;
            ahrs->accelerometerIgnored = false;
            ahrs->accelerationRejectionTimer -= ahrs->accelerationRejectionTimer >= 10 ? 10 : 0;
        } else {
            ahrs->accelerationRejectionTimer++;
        }
    }

    // Calculate magnetometer feedback
    FusionVector halfMagnetometerFeedback = FUSION_VECTOR_ZERO;
    ahrs->magnetometerIgnored = true;
    if (FusionVectorIsZero(magnetometer) == false) {

        // Set to compass heading if magnetic rejection times out
        ahrs->magneticRejectionTimeout = false;
        if (ahrs->magneticRejectionTimer > ahrs->settings.rejectionTimeout) {
            FusionAhrsSetHeading(ahrs, FusionCompassCalculateHeading(ahrs->settings.convention, halfGravity, magnetometer));
            ahrs->magneticRejectionTimer = 0;
            ahrs->magneticRejectionTimeout = true;
        }

        // Calculate direction of magnetic field indicated by algorithm
        const FusionVector halfMagnetic = HalfMagnetic(ahrs);

        // Calculate magnetometer feedback scaled by 0.5
        ahrs->halfMagnetometerFeedback = FusionVectorCrossProduct(FusionVectorNormalise(FusionVectorCrossProduct(halfGravity, magnetometer)), halfMagnetic);

        // Ignore magnetometer if magnetic distortion detected
        if ((ahrs->initialising == true) || (FusionVectorMagnitudeSquared(ahrs->halfMagnetometerFeedback) <= ahrs->settings.magneticRejection)) {
            halfMagnetometerFeedback = ahrs->halfMagnetometerFeedback;
            ahrs->magnetometerIgnored = false;
            ahrs->magneticRejectionTimer -= ahrs->magneticRejectionTimer >= 10 ? 10 : 0;
        } else {
            ahrs->magneticRejectionTimer++;
        }
    }

    // Convert gyroscope to radians per second scaled by 0.5
    const FusionVector halfGyroscope = FusionVectorMultiplyScalar(gyroscope, FusionDegreesToRadians(0.5f));

    // Apply feedback to gyroscope
    const FusionVector adjustedHalfGyroscope = FusionVectorAdd(halfGyroscope, FusionVectorMultiplyScalar(FusionVectorAdd(halfAccelerometerFeedback, halfMagnetometerFeedback), ahrs->rampedGain));

    // Integrate rate of change of quaternion
    ahrs->quaternion = FusionQuaternionAdd(ahrs->quaternion, FusionQuaternionMultiplyVector(ahrs->quaternion, FusionVectorMultiplyScalar(adjustedHalfGyroscope, deltaTime)));

    // Normalise quaternion
    ahrs->quaternion = FusionQuaternionNormalise(ahrs->quaternion);
#undef Q
}

/**
 * @brief Returns the direction of gravity scaled by 0.5.
 * @param ahrs AHRS algorithm structure.
 * @return Direction of gravity scaled by 0.5.
 */
static FusionVector HalfGravity(const FusionAhrs *const ahrs) {
#define Q ahrs->quaternion.element
    switch (ahrs->settings.convention) {
        case FusionConventionNwu:
        case FusionConventionEnu: {
            const FusionVector halfGravity = {.axis = {
                    .x = Q.x * Q.z - Q.w * Q.y,
                    .y = Q.y * Q.z + Q.w * Q.x,
                    .z = Q.w * Q.w - 0.5f + Q.z * Q.z,
            }}; // third column of transposed rotation matrix scaled by 0.5
            return halfGravity;
        }
        case FusionConventionNed: {
            const FusionVector halfGravity = {.axis = {
                    .x = Q.w * Q.y - Q.x * Q.z,
                    .y = -1.0f * (Q.y * Q.z + Q.w * Q.x),
                    .z = 0.5f - Q.w * Q.w - Q.z * Q.z,
            }}; // third column of transposed rotation matrix scaled by -0.5
            return halfGravity;
        }
    }
    return FUSION_VECTOR_ZERO; // avoid compiler warning
#undef Q
}

/**
 * @brief Returns the direction of the magnetic field scaled by 0.5.
 * @param ahrs AHRS algorithm structure.
 * @return Direction of the magnetic field scaled by 0.5.
 */
static FusionVector HalfMagnetic(const FusionAhrs *const ahrs) {
#define Q ahrs->quaternion.element
    switch (ahrs->settings.convention) {
        case FusionConventionNwu: {
            const FusionVector halfMagnetic = {.axis = {
                    .x = Q.x * Q.y + Q.w * Q.z,
                    .y = Q.w * Q.w - 0.5f + Q.y * Q.y,
                    .z = Q.y * Q.z - Q.w * Q.x,
            }}; // second column of transposed rotation matrix scaled by 0.5
            return halfMagnetic;
        }
        case FusionConventionEnu: {
            const FusionVector halfMagnetic = {.axis = {
                    .x = 0.5f - Q.w * Q.w - Q.x * Q.x,
                    .y = Q.w * Q.z - Q.x * Q.y,
                    .z = -1.0f * (Q.x * Q.z + Q.w * Q.y),
            }}; // first column of transposed rotation matrix scaled by -0.5
            return halfMagnetic;
        }
        case FusionConventionNed: {
            const FusionVector halfMagnetic = {.axis = {
                    .x = -1.0f * (Q.x * Q.y + Q.w * Q.z),
                    .y = 0.5f - Q.w * Q.w - Q.y * Q.y,
                    .z = Q.w * Q.x - Q.y * Q.z,
            }}; // second column of transposed rotation matrix scaled by -0.5
            return halfMagnetic;
        }
    }
    return FUSION_VECTOR_ZERO; // avoid compiler warning
#undef Q
}

/**
 * @brief Updates the AHRS algorithm using the gyroscope and accelerometer
 * measurements only.
 * @param ahrs AHRS algorithm structure.
 * @param gyroscope Gyroscope measurement in degrees per second.
 * @param accelerometer Accelerometer measurement in g.
 * @param deltaTime Delta time in seconds.
 */
void FusionAhrsUpdateNoMagnetometer(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const float deltaTime) {

    // Update AHRS algorithm
    FusionAhrsUpdate(ahrs, gyroscope, accelerometer, FUSION_VECTOR_ZERO, deltaTime);

    // Zero heading during initialisation
    if ((ahrs->initialising == true) && (ahrs->accelerationRejectionTimeout == false)) {
        FusionAhrsSetHeading(ahrs, 0.0f);
    }
}

/**
 * @brief Updates the AHRS algorithm using the gyroscope, accelerometer, and
 * heading measurements.
 * @param ahrs AHRS algorithm structure.
 * @param gyroscope Gyroscope measurement in degrees per second.
 * @param accelerometer Accelerometer measurement in g.
 * @param heading Heading measurement in degrees.
 * @param deltaTime Delta time in seconds.
 */
void FusionAhrsUpdateExternalHeading(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const float heading, const float deltaTime) {
#define Q ahrs->quaternion.element

    // Calculate roll
    const float roll = atan2f(Q.w * Q.x + Q.y * Q.z, 0.5f - Q.y * Q.y - Q.x * Q.x);

    // Calculate magnetometer
    const float headingRadians = FusionDegreesToRadians(heading);
    const float sinHeadingRadians = sinf(headingRadians);
    const FusionVector magnetometer = {.axis = {
            .x = cosf(headingRadians),
            .y = -1.0f * cosf(roll) * sinHeadingRadians,
            .z = sinHeadingRadians * sinf(roll),
    }};

    // Update AHRS algorithm
    FusionAhrsUpdate(ahrs, gyroscope, accelerometer, magnetometer, deltaTime);
#undef Q
}

/**
 * @brief Returns the quaternion describing the sensor relative to the Earth.
 * @param ahrs AHRS algorithm structure.
 * @return Quaternion describing the sensor relative to the Earth.
 */
FusionQuaternion FusionAhrsGetQuaternion(const FusionAhrs *const ahrs) {
    return ahrs->quaternion;
}

/**
 * @brief Returns the linear acceleration measurement equal to the accelerometer
 * measurement with the 1 g of gravity removed.
 * @param ahrs AHRS algorithm structure.
 * @return Linear acceleration measurement in g.
 */
FusionVector FusionAhrsGetLinearAcceleration(const FusionAhrs *const ahrs) {
#define Q ahrs->quaternion.element

    // Calculate gravity in the sensor coordinate frame
    const FusionVector gravity = {.axis = {
            .x = 2.0f * (Q.x * Q.z - Q.w * Q.y),
            .y = 2.0f * (Q.y * Q.z + Q.w * Q.x),
            .z = 2.0f * (Q.w * Q.w - 0.5f + Q.z * Q.z),
    }}; // third column of transposed rotation matrix

    // Remove gravity from accelerometer measurement
    switch (ahrs->settings.convention) {
        case FusionConventionNwu:
        case FusionConventionEnu: {
            return FusionVectorSubtract(ahrs->accelerometer, gravity);
        }
        case FusionConventionNed: {
            return FusionVectorAdd(ahrs->accelerometer, gravity);
        }
    }
    return FUSION_VECTOR_ZERO; // avoid compiler warning
#undef Q
}

/**
 * @brief Returns the Earth acceleration measurement equal to accelerometer
 * measurement in the Earth coordinate frame with the 1 g of gravity removed.
 * @param ahrs AHRS algorithm structure.
 * @return Earth acceleration measurement in g.
 */
FusionVector FusionAhrsGetEarthAcceleration(const FusionAhrs *const ahrs) {
#define Q ahrs->quaternion.element
#define A ahrs->accelerometer.axis

    // Calculate accelerometer measurement in the Earth coordinate frame
    const float qwqw = Q.w * Q.w; // calculate common terms to avoid repeated operations
    const float qwqx = Q.w * Q.x;
    const float qwqy = Q.w * Q.y;
    const float qwqz = Q.w * Q.z;
    const float qxqy = Q.x * Q.y;
    const float qxqz = Q.x * Q.z;
    const float qyqz = Q.y * Q.z;
    FusionVector accelerometer = {.axis = {
            .x = 2.0f * ((qwqw - 0.5f + Q.x * Q.x) * A.x + (qxqy - qwqz) * A.y + (qxqz + qwqy) * A.z),
            .y = 2.0f * ((qxqy + qwqz) * A.x + (qwqw - 0.5f + Q.y * Q.y) * A.y + (qyqz - qwqx) * A.z),
            .z = 2.0f * ((qxqz - qwqy) * A.x + (qyqz + qwqx) * A.y + (qwqw - 0.5f + Q.z * Q.z) * A.z),
    }}; // rotation matrix multiplied with the accelerometer

    // Remove gravity from accelerometer measurement
    switch (ahrs->settings.convention) {
        case FusionConventionNwu:
        case FusionConventionEnu:
            accelerometer.axis.z -= 1.0f;
            break;
        case FusionConventionNed:
            accelerometer.axis.z += 1.0f;
            break;
    }
    return accelerometer;
#undef Q
#undef A
}

/**
 * @brief Returns the AHRS algorithm internal states.
 * @param ahrs AHRS algorithm structure.
 * @return AHRS algorithm internal states.
 */
FusionAhrsInternalStates FusionAhrsGetInternalStates(const FusionAhrs *const ahrs) {
    const FusionAhrsInternalStates internalStates = {
            .accelerationError = FusionRadiansToDegrees(FusionAsin(2.0f * FusionVectorMagnitude(ahrs->halfAccelerometerFeedback))),
            .accelerometerIgnored = ahrs->accelerometerIgnored,
            .accelerationRejectionTimer = ahrs->settings.rejectionTimeout == 0 ? 0.0f : (float) ahrs->accelerationRejectionTimer / (float) ahrs->settings.rejectionTimeout,
            .magneticError = FusionRadiansToDegrees(FusionAsin(2.0f * FusionVectorMagnitude(ahrs->halfMagnetometerFeedback))),
            .magnetometerIgnored = ahrs->magnetometerIgnored,
            .magneticRejectionTimer = ahrs->settings.rejectionTimeout == 0 ? 0.0f : (float) ahrs->magneticRejectionTimer / (float) ahrs->settings.rejectionTimeout,
    };
    return internalStates;
}

/**
 * @brief Returns the AHRS algorithm flags.
 * @param ahrs AHRS algorithm structure.
 * @return AHRS algorithm flags.
 */
FusionAhrsFlags FusionAhrsGetFlags(const FusionAhrs *const ahrs) {
    const unsigned int warningTimeout = ahrs->settings.rejectionTimeout / 4;
    const FusionAhrsFlags flags = {
            .initialising = ahrs->initialising,
            .accelerationRejectionWarning = ahrs->accelerationRejectionTimer > warningTimeout,
            .accelerationRejectionTimeout = ahrs->accelerationRejectionTimeout,
            .magneticRejectionWarning = ahrs->magneticRejectionTimer > warningTimeout,
            .magneticRejectionTimeout = ahrs->magneticRejectionTimeout,
    };
    return flags;
}

/**
 * @brief Sets the heading of the orientation measurement provided by the AHRS
 * algorithm.  This function can be used to reset drift in heading when the AHRS
 * algorithm is being used without a magnetometer.
 * @param ahrs AHRS algorithm structure.
 * @param heading Heading angle in degrees.
 */
void FusionAhrsSetHeading(FusionAhrs *const ahrs, const float heading) {
#define Q ahrs->quaternion.element
    const float yaw = atan2f(Q.w * Q.z + Q.x * Q.y, 0.5f - Q.y * Q.y - Q.z * Q.z);
    const float halfYawMinusHeading = 0.5f * (yaw - FusionDegreesToRadians(heading));
    const FusionQuaternion rotation = {.element = {
            .w = cosf(halfYawMinusHeading),
            .x = 0.0f,
            .y = 0.0f,
            .z = -1.0f * sinf(halfYawMinusHeading),
    }};
    ahrs->quaternion = FusionQuaternionMultiply(rotation, ahrs->quaternion);
#undef Q
}

//------------------------------------------------------------------------------
// End of file
