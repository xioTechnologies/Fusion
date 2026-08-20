/**
 * @file FusionAhrs.c
 * @author Seb Madgwick
 * @brief Attitude and Heading Reference System (AHRS) algorithm.
 */

//------------------------------------------------------------------------------
// Includes

#include <float.h>
#include "FusionAhrs.h"
#include "FusionInline.h"
#include <math.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Initial startup gain.
 */
#define INITIAL_STARTUP_GAIN (10.0f)

/**
 * @brief Startup period in seconds.
 */
#define STARTUP_PERIOD (3.0f)

//------------------------------------------------------------------------------
// Function declarations

static FUSION_INLINE void Overrange(FusionAhrs *const ahrs, const FusionVector gyroscope);

static FUSION_INLINE void SoftRestart(FusionAhrs *const ahrs);

static FUSION_INLINE float Startup(FusionAhrs *const ahrs);

static FUSION_INLINE FusionVector HalfInclinationFeedback(FusionAhrs *const ahrs, const FusionVector halfGravity, const FusionVector accelerometer);

static FUSION_INLINE FusionVector HalfHeadingFeedback(FusionAhrs *const ahrs, const FusionVector halfGravity, const FusionVector magnetometer);

static FUSION_INLINE FusionVector HalfGravity(const FusionAhrs *const ahrs);

static FUSION_INLINE FusionVector HalfWest(const FusionAhrs *const ahrs);

static FUSION_INLINE FusionVector Residual(const FusionVector sensor, const FusionVector reference);

static FUSION_INLINE int32_t Clamp(const int32_t value, const int32_t min, const int32_t max);

//------------------------------------------------------------------------------
// Variables

const FusionAhrsSettings fusionAhrsDefaultSettings = {
    .sampleRate = 100.0f,
    .convention = FusionConventionNwu,
    .gain = 0.5f,
    .gyroscopeRange = 0.0f,
    .accelerationRejection = 0.0f,
    .magneticRejection = 0.0f,
    .rejectionTimeout = 0.0f,
};

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the AHRS structure.
 * @param ahrs AHRS structure.
 */
void FusionAhrsInitialise(FusionAhrs *const ahrs) {
    FusionAhrsSetSettings(ahrs, &fusionAhrsDefaultSettings);
    FusionAhrsRestart(ahrs);
}

/**
 * @brief Sets the settings.
 * @param ahrs AHRS structure.
 * @param settings Settings.
 */
void FusionAhrsSetSettings(FusionAhrs *const ahrs, const FusionAhrsSettings *const settings) {
    ahrs->samplePeriod = 1.0f / settings->sampleRate;
    ahrs->convention = settings->convention;

    ahrs->gain = settings->gain;
    ahrs->startupGainRate = ((INITIAL_STARTUP_GAIN - ahrs->gain) / STARTUP_PERIOD) * ahrs->samplePeriod;

    ahrs->overrangeEnabled = settings->gyroscopeRange > 0.0f;
    ahrs->overrangeThreshold = 0.98f * settings->gyroscopeRange;

    ahrs->accelerationRejection = settings->accelerationRejection == 0.0f ? FLT_MAX : powf(0.5f * sinf(FusionDegreesToRadians(settings->accelerationRejection)), 2);
    ahrs->magneticRejection = settings->magneticRejection == 0.0f ? FLT_MAX : powf(0.5f * sinf(FusionDegreesToRadians(settings->magneticRejection)), 2);
    ahrs->rejectionTimeout = (int32_t) (settings->sampleRate * settings->rejectionTimeout);

    ahrs->accelerationRecoveryThreshold = ahrs->rejectionTimeout;
    ahrs->magneticRecoveryThreshold = ahrs->rejectionTimeout;

    if ((settings->gain == 0.0f) || (settings->rejectionTimeout == 0.0f)) {
        ahrs->accelerationRejection = FLT_MAX; // disable acceleration and magnetic rejection features if gain is zero
        ahrs->magneticRejection = FLT_MAX;
    }
}

/**
 * @brief Sets the sample period. The sample period must be approximately equal
 * to the current settings. This function is intended to be called before each
 * algorithm update to compensate for gyroscope sample clock errors.
 * @param ahrs AHRS structure.
 * @param samplePeriod Sample period in seconds.
 */
void FusionAhrsSetSamplePeriod(FusionAhrs *const ahrs, const float samplePeriod) {
    ahrs->samplePeriod = samplePeriod;
}

/**
 * @brief Restarts the AHRS algorithm.
 * @param ahrs AHRS structure.
 */
void FusionAhrsRestart(FusionAhrs *const ahrs) {
    ahrs->quaternion = FUSION_QUATERNION_IDENTITY;
    ahrs->accelerometer = FUSION_VECTOR_ZERO;

    ahrs->startup = true;
    ahrs->startupGain = INITIAL_STARTUP_GAIN;

    ahrs->overrangeRecovery = false;

    ahrs->halfAccelerometerResidual = FUSION_VECTOR_ZERO;
    ahrs->accelerationRecoveryTrigger = 0;
    ahrs->accelerationRecoveryThreshold = ahrs->rejectionTimeout;
    ahrs->accelerometerIgnored = false;

    ahrs->halfMagnetometerResidual = FUSION_VECTOR_ZERO;
    ahrs->magneticRecoveryTrigger = 0;
    ahrs->magneticRecoveryThreshold = ahrs->rejectionTimeout;
    ahrs->magnetometerIgnored = false;
}

/**
 * @brief Skips startup. This function is intended to be called before the
 * first algorithm update when the initial orientation is already known.
 * @param ahrs AHRS structure.
 */
void FusionAhrsSkipStartup(FusionAhrs *const ahrs) {
    ahrs->startup = false;
    ahrs->overrangeRecovery = false;
}

/**
 * @brief Updates the AHRS algorithm using the gyroscope, accelerometer, and
 * magnetometer.
 * @param ahrs AHRS structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @param accelerometer Accelerometer in g.
 * @param magnetometer Magnetometer in any calibrated units.
 */
void FusionAhrsUpdate(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const FusionVector magnetometer) {
    ahrs->accelerometer = accelerometer;

    Overrange(ahrs, gyroscope);

    const float gain = Startup(ahrs);

    const FusionVector halfGyroscope = FusionVectorScale(gyroscope, FusionDegreesToRadians(0.5f));

    const FusionVector halfGravity = HalfGravity(ahrs);

    const FusionVector halfFeedback = FusionVectorAdd(HalfInclinationFeedback(ahrs, halfGravity, accelerometer), HalfHeadingFeedback(ahrs, halfGravity, magnetometer));

    const FusionVector halfAngularRate = FusionVectorAdd(halfGyroscope, FusionVectorScale(halfFeedback, gain));

    ahrs->quaternion = FusionQuaternionAdd(ahrs->quaternion, FusionQuaternionVectorProduct(ahrs->quaternion, FusionVectorScale(halfAngularRate, ahrs->samplePeriod)));

    ahrs->quaternion = FusionQuaternionNormalise(ahrs->quaternion);
}

/**
 * @brief Triggers soft restart if overrange detected.
 * @param ahrs AHRS structure.
 * @param gyroscope Gyroscope in degrees per second.
 */
static FUSION_INLINE void Overrange(FusionAhrs *const ahrs, const FusionVector gyroscope) {
    if (ahrs->overrangeEnabled == false) {
        return;
    }

    if ((fabsf(gyroscope.axis.x) <= ahrs->overrangeThreshold) &&
        (fabsf(gyroscope.axis.y) <= ahrs->overrangeThreshold) &&
        (fabsf(gyroscope.axis.z) <= ahrs->overrangeThreshold)) {
        return;
    }

    SoftRestart(ahrs);
    ahrs->overrangeRecovery = true;
}

/**
 * @brief Restarts the AHRS algorithm while preserving outputs.
 * @param ahrs AHRS structure.
 */
static FUSION_INLINE void SoftRestart(FusionAhrs *const ahrs) {
    const FusionQuaternion quaternion = ahrs->quaternion;
    const FusionVector accelerometer = ahrs->accelerometer;

    FusionAhrsRestart(ahrs);

    ahrs->quaternion = quaternion;
    ahrs->accelerometer = accelerometer;
}

/**
 * @brief Ramps down the gain during startup.
 * @param ahrs AHRS structure.
 * @return Gain.
 */
static FUSION_INLINE float Startup(FusionAhrs *const ahrs) {
    if (ahrs->startup == false) {
        return ahrs->gain;
    }

    ahrs->startupGain -= ahrs->startupGainRate;

    if (ahrs->startupGain > ahrs->gain) {
        return ahrs->startupGain;
    }

    ahrs->startup = false;
    ahrs->overrangeRecovery = false;

    return ahrs->gain;
}

/**
 * @brief Returns inclination feedback scaled by 0.5.
 * @param ahrs AHRS structure.
 * @param halfGravity Direction of gravity scaled by 0.5.
 * @param accelerometer Accelerometer in g.
 * @return Inclination feedback scaled by 0.5.
 */
static FUSION_INLINE FusionVector HalfInclinationFeedback(FusionAhrs *const ahrs, const FusionVector halfGravity, const FusionVector accelerometer) {
    FusionVector halfInclinationFeedback = FUSION_VECTOR_ZERO;
    ahrs->accelerometerIgnored = true;
    if (FusionVectorIsZero(accelerometer) == false) {
        // Calculate accelerometer residual scaled by 0.5
        ahrs->halfAccelerometerResidual = Residual(FusionVectorNormalise(accelerometer), halfGravity);

        // Don't ignore accelerometer if acceleration error below threshold
        if (ahrs->startup || (FusionVectorNormSquared(ahrs->halfAccelerometerResidual) <= ahrs->accelerationRejection)) {
            ahrs->accelerometerIgnored = false;
            ahrs->accelerationRecoveryTrigger -= 9;
        } else {
            ahrs->accelerationRecoveryTrigger += 1;
        }

        // Don't ignore accelerometer during acceleration recovery
        if (ahrs->accelerationRecoveryTrigger > ahrs->accelerationRecoveryThreshold) {
            ahrs->accelerationRecoveryThreshold = 0;
            ahrs->accelerometerIgnored = false;
        } else {
            ahrs->accelerationRecoveryThreshold = ahrs->rejectionTimeout;
        }
        ahrs->accelerationRecoveryTrigger = Clamp(ahrs->accelerationRecoveryTrigger, 0, ahrs->rejectionTimeout);

        // Apply accelerometer feedback
        if (ahrs->accelerometerIgnored == false) {
            halfInclinationFeedback = ahrs->halfAccelerometerResidual;
        }
    }
    return halfInclinationFeedback;
}

/**
 * @brief Returns heading feedback scaled by 0.5.
 * @param ahrs AHRS structure.
 * @param halfGravity Direction of gravity scaled by 0.5.
 * @param magnetometer Magnetometer in any calibrated units.
 * @return Heading feedback scaled by 0.5.
 */
static FUSION_INLINE FusionVector HalfHeadingFeedback(FusionAhrs *const ahrs, const FusionVector halfGravity, const FusionVector magnetometer) {
    FusionVector halfHeadingFeedback = FUSION_VECTOR_ZERO;
    ahrs->magnetometerIgnored = true;
    if (FusionVectorIsZero(magnetometer) == false) {
        // Calculate direction of magnetic field indicated by algorithm
        const FusionVector halfWest = HalfWest(ahrs);

        // Calculate magnetometer residual scaled by 0.5
        ahrs->halfMagnetometerResidual = Residual(FusionVectorNormalise(FusionVectorCross(halfGravity, magnetometer)), halfWest);

        // Don't ignore magnetometer if magnetic error below threshold
        if (ahrs->startup || (FusionVectorNormSquared(ahrs->halfMagnetometerResidual) <= ahrs->magneticRejection)) {
            ahrs->magnetometerIgnored = false;
            ahrs->magneticRecoveryTrigger -= 9;
        } else {
            ahrs->magneticRecoveryTrigger += 1;
        }

        // Don't ignore magnetometer during magnetic recovery
        if (ahrs->magneticRecoveryTrigger > ahrs->magneticRecoveryThreshold) {
            ahrs->magneticRecoveryThreshold = 0;
            ahrs->magnetometerIgnored = false;
        } else {
            ahrs->magneticRecoveryThreshold = ahrs->rejectionTimeout;
        }
        ahrs->magneticRecoveryTrigger = Clamp(ahrs->magneticRecoveryTrigger, 0, ahrs->rejectionTimeout);

        // Apply magnetometer feedback
        if (ahrs->magnetometerIgnored == false) {
            halfHeadingFeedback = ahrs->halfMagnetometerResidual;
        }
    }
    return halfHeadingFeedback;
}

/**
 * @brief Returns the direction of gravity scaled by 0.5.
 * @param ahrs AHRS structure.
 * @return Direction of gravity scaled by 0.5.
 */
static FUSION_INLINE FusionVector HalfGravity(const FusionAhrs *const ahrs) {
#define Q ahrs->quaternion.element
    switch (ahrs->convention) {
        case FusionConventionNwu:
        case FusionConventionEnu: {
            const FusionVector halfGravity = {
                .axis = {
                    .x = Q.x * Q.z - Q.w * Q.y,
                    .y = Q.y * Q.z + Q.w * Q.x,
                    .z = Q.w * Q.w - 0.5f + Q.z * Q.z,
                }
            }; // third column of transposed rotation matrix scaled by 0.5
            return halfGravity;
        }
        case FusionConventionNed: {
            const FusionVector halfGravity = {
                .axis = {
                    .x = Q.w * Q.y - Q.x * Q.z,
                    .y = -1.0f * (Q.y * Q.z + Q.w * Q.x),
                    .z = 0.5f - Q.w * Q.w - Q.z * Q.z,
                }
            }; // third column of transposed rotation matrix scaled by -0.5
            return halfGravity;
        }
    }
#undef Q
    return FUSION_VECTOR_ZERO; // avoid compiler warning
}

/**
 * @brief Returns the direction of west scaled by 0.5. The cross product of
 * gravity and the magnetometer is west.
 * @param ahrs AHRS structure.
 * @return Direction of west scaled by 0.5.
 */
static FUSION_INLINE FusionVector HalfWest(const FusionAhrs *const ahrs) {
#define Q ahrs->quaternion.element
    switch (ahrs->convention) {
        case FusionConventionNwu: {
            const FusionVector halfWest = {
                .axis = {
                    .x = Q.x * Q.y + Q.w * Q.z,
                    .y = Q.w * Q.w - 0.5f + Q.y * Q.y,
                    .z = Q.y * Q.z - Q.w * Q.x,
                }
            }; // second column of transposed rotation matrix scaled by 0.5
            return halfWest;
        }
        case FusionConventionEnu: {
            const FusionVector halfWest = {
                .axis = {
                    .x = 0.5f - Q.w * Q.w - Q.x * Q.x,
                    .y = Q.w * Q.z - Q.x * Q.y,
                    .z = -1.0f * (Q.x * Q.z + Q.w * Q.y),
                }
            }; // first column of transposed rotation matrix scaled by -0.5
            return halfWest;
        }
        case FusionConventionNed: {
            const FusionVector halfWest = {
                .axis = {
                    .x = -1.0f * (Q.x * Q.y + Q.w * Q.z),
                    .y = 0.5f - Q.w * Q.w - Q.y * Q.y,
                    .z = Q.w * Q.x - Q.y * Q.z,
                }
            }; // second column of transposed rotation matrix scaled by -0.5
            return halfWest;
        }
    }
#undef Q
    return FUSION_VECTOR_ZERO; // avoid compiler warning
}

/**
 * @brief Returns the residual between the sensor and reference vector.
 * @param sensor Sensor.
 * @param reference Reference.
 * @return Residual between the sensor and reference vector.
 */
static FUSION_INLINE FusionVector Residual(const FusionVector sensor, const FusionVector reference) {
    if (FusionVectorDot(sensor, reference) > 0.0f) {
        return FusionVectorCross(sensor, reference); // if error is <90 degrees
    }

    const FusionVector cross = FusionVectorCross(sensor, reference);

    if (FusionVectorIsZero(cross)) {
        return FUSION_VECTOR_ZERO;
    }

    return FusionVectorNormalise(cross);
}

/**
 * @brief Returns a value limited to maximum and minimum.
 * @param value Value.
 * @param min Minimum value.
 * @param max Maximum value.
 * @return Value limited to maximum and minimum.
 */
static FUSION_INLINE int32_t Clamp(const int32_t value, const int32_t min, const int32_t max) {
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
}

/**
 * @brief Updates the AHRS algorithm using the gyroscope and accelerometer.
 * @param ahrs AHRS structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @param accelerometer Accelerometer in g.
 */
void FusionAhrsUpdateNoMagnetometer(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer) {
    FusionAhrsUpdate(ahrs, gyroscope, accelerometer, FUSION_VECTOR_ZERO);

    // Zero heading during startup
    if (ahrs->startup) {
        FusionAhrsSetHeading(ahrs, 0.0f);
    }
}

/**
 * @brief Updates the AHRS algorithm using the gyroscope, accelerometer, and an
 * external measurement of heading.
 * @param ahrs AHRS structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @param accelerometer Accelerometer in g.
 * @param heading Heading in degrees.
 */
void FusionAhrsUpdateExternalHeading(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const float heading) {
#define Q ahrs->quaternion.element
    const float roll = atan2f(Q.w * Q.x + Q.y * Q.z, 0.5f - Q.y * Q.y - Q.x * Q.x);
#undef Q

    // Calculate equivalent magnetometer
    const float headingRadians = FusionDegreesToRadians(heading);
    const float sinHeadingRadians = sinf(headingRadians);
    const FusionVector magnetometer = {
        .axis = {
            .x = cosf(headingRadians),
            .y = -1.0f * cosf(roll) * sinHeadingRadians,
            .z = sinHeadingRadians * sinf(roll),
        }
    };

    // Update algorithm
    FusionAhrsUpdate(ahrs, gyroscope, accelerometer, magnetometer);
}

/**
 * @brief Returns the quaternion.
 * @param ahrs AHRS structure.
 * @return Quaternion.
 */
FusionQuaternion FusionAhrsGetQuaternion(const FusionAhrs *const ahrs) {
    return ahrs->quaternion;
}

/**
 * @brief Sets the quaternion.
 * @param ahrs AHRS structure.
 * @param quaternion Quaternion.
 */
void FusionAhrsSetQuaternion(FusionAhrs *const ahrs, const FusionQuaternion quaternion) {
    ahrs->quaternion = quaternion;
}

/**
 * @brief Returns the direction of gravity.
 * @param ahrs AHRS structure.
 * @return Direction of gravity as a unit vector.
 */
FusionVector FusionAhrsGetGravity(const FusionAhrs *const ahrs) {
    return FusionVectorScale(HalfGravity(ahrs), 2.0f);
}

/**
 * @brief Returns the linear acceleration.
 * @param ahrs AHRS structure.
 * @return Linear acceleration in g.
 */
FusionVector FusionAhrsGetLinearAcceleration(const FusionAhrs *const ahrs) {
    return FusionVectorSubtract(ahrs->accelerometer, FusionAhrsGetGravity(ahrs));
}

/**
 * @brief Returns the Earth acceleration.
 * @param ahrs AHRS structure.
 * @return Earth acceleration in g.
 */
FusionVector FusionAhrsGetEarthAcceleration(const FusionAhrs *const ahrs) {
#define Q ahrs->quaternion.element
#define A ahrs->accelerometer.axis
    FusionVector acceleration = {
        .axis = {
            .x = 2.0f * ((Q.w * Q.w - 0.5f + Q.x * Q.x) * A.x + (Q.x * Q.y - Q.w * Q.z) * A.y + (Q.x * Q.z + Q.w * Q.y) * A.z),
            .y = 2.0f * ((Q.x * Q.y + Q.w * Q.z) * A.x + (Q.w * Q.w - 0.5f + Q.y * Q.y) * A.y + (Q.y * Q.z - Q.w * Q.x) * A.z),
            .z = 2.0f * ((Q.x * Q.z - Q.w * Q.y) * A.x + (Q.y * Q.z + Q.w * Q.x) * A.y + (Q.w * Q.w - 0.5f + Q.z * Q.z) * A.z),
        }
    }; // rotation matrix multiplied with the accelerometer
#undef Q
#undef A

    switch (ahrs->convention) {
        case FusionConventionNwu:
        case FusionConventionEnu:
            acceleration.axis.z -= 1.0f;
            break;
        case FusionConventionNed:
            acceleration.axis.z += 1.0f;
            break;
    }
    return acceleration;
}

/**
 * @brief Returns the internal states.
 * @param ahrs AHRS structure.
 * @return Internal states.
 */
FusionAhrsInternalStates FusionAhrsGetInternalStates(const FusionAhrs *const ahrs) {
    const FusionAhrsInternalStates internalStates = {
        .accelerationError = FusionRadiansToDegrees(FusionArcSin(2.0f * FusionVectorNorm(ahrs->halfAccelerometerResidual))),
        .accelerometerIgnored = ahrs->accelerometerIgnored,
        .accelerationRecoveryTrigger = ahrs->rejectionTimeout == 0 ? 0.0f : (float) ahrs->accelerationRecoveryTrigger / (float) ahrs->rejectionTimeout,
        .magneticError = FusionRadiansToDegrees(FusionArcSin(2.0f * FusionVectorNorm(ahrs->halfMagnetometerResidual))),
        .magnetometerIgnored = ahrs->magnetometerIgnored,
        .magneticRecoveryTrigger = ahrs->rejectionTimeout == 0 ? 0.0f : (float) ahrs->magneticRecoveryTrigger / (float) ahrs->rejectionTimeout,
    };
    return internalStates;
}

/**
 * @brief Returns the flags.
 * @param ahrs AHRS structure.
 * @return Flags.
 */
FusionAhrsFlags FusionAhrsGetFlags(const FusionAhrs *const ahrs) {
    const FusionAhrsFlags flags = {
        .startup = ahrs->startup,
        .overrangeRecovery = ahrs->overrangeRecovery,
        .accelerationRecovery = ahrs->accelerationRecoveryTrigger > ahrs->accelerationRecoveryThreshold,
        .magneticRecovery = ahrs->magneticRecoveryTrigger > ahrs->magneticRecoveryThreshold,
    };
    return flags;
}

/**
 * @brief Sets the heading.
 * @param ahrs AHRS structure.
 * @param heading Heading in degrees.
 */
void FusionAhrsSetHeading(FusionAhrs *const ahrs, const float heading) {
#define Q ahrs->quaternion.element
    const float yaw = atan2f(Q.w * Q.z + Q.x * Q.y, 0.5f - Q.y * Q.y - Q.z * Q.z);
#undef Q
    const float halfYawMinusHeading = 0.5f * (yaw - FusionDegreesToRadians(heading));

    const FusionQuaternion rotation = {
        .element = {
            .w = cosf(halfYawMinusHeading),
            .x = 0.0f,
            .y = 0.0f,
            .z = -1.0f * sinf(halfYawMinusHeading),
        }
    };

    ahrs->quaternion = FusionQuaternionProduct(rotation, ahrs->quaternion);
}

//------------------------------------------------------------------------------
// End of file
