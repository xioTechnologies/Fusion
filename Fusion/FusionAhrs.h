/**
 * @file FusionAhrs.h
 * @author Seb Madgwick
 * @brief Attitude and Heading Reference System (AHRS) algorithm.
 */

#ifndef FUSION_AHRS_H
#define FUSION_AHRS_H

//------------------------------------------------------------------------------
// Includes

#include "FusionConvention.h"
#include "FusionMath.h"
#include <stdbool.h>
#include <stdint.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Settings.
 */
typedef struct {
    float sampleRate; // Hz
    FusionConvention convention;
    float gain;
    float gyroscopeRange; // degrees per second
    float accelerationRejection; // degrees
    float magneticRejection; // degrees
    float rejectionTimeout; // seconds
} FusionAhrsSettings;

/**
 * @brief AHRS structure. All members are private.
 */
typedef struct {
    // Settings
    float samplePeriod;
    FusionConvention convention;
    float gain;
    float gyroscopeRange;
    float accelerationRejection;
    float magneticRejection;
    int32_t rejectionTimeout;

    // Outputs
    FusionQuaternion quaternion;
    FusionVector accelerometer;
    FusionVector halfGravity;

    // Startup
    bool startup;
    float rampedGain;
    float rampedGainStep;

    // Gyroscope overrange
    bool angularRateRecovery;

    // Acceleration and magnetic rejection
    FusionVector halfAccelerometerFeedback;
    FusionVector halfMagnetometerFeedback;
    bool accelerometerIgnored;
    int32_t accelerationRecoveryTrigger;
    int32_t accelerationRecoveryThreshold;
    bool magnetometerIgnored;
    int32_t magneticRecoveryTrigger;
    int32_t magneticRecoveryThreshold;
} FusionAhrs;

/**
 * @brief Internal states.
 */
typedef struct {
    float accelerationError;
    bool accelerometerIgnored;
    float accelerationRecoveryTrigger;
    float magneticError;
    bool magnetometerIgnored;
    float magneticRecoveryTrigger;
} FusionAhrsInternalStates;

/**
 * @brief Flags.
 */
typedef struct {
    bool startup;
    bool angularRateRecovery;
    bool accelerationRecovery;
    bool magneticRecovery;
} FusionAhrsFlags;

//------------------------------------------------------------------------------
// Variable declarations

extern const FusionAhrsSettings fusionAhrsDefaultSettings;

//------------------------------------------------------------------------------
// Function declarations

void FusionAhrsInitialise(FusionAhrs *const ahrs);

void FusionAhrsRestart(FusionAhrs *const ahrs);

void FusionAhrsSetSettings(FusionAhrs *const ahrs, const FusionAhrsSettings *const settings);

void FusionAhrsSetSamplePeriod(FusionAhrs *const ahrs, const float samplePeriod);

void FusionAhrsUpdate(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const FusionVector magnetometer);

void FusionAhrsUpdateNoMagnetometer(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer);

void FusionAhrsUpdateExternalHeading(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const float heading);

FusionQuaternion FusionAhrsGetQuaternion(const FusionAhrs *const ahrs);

void FusionAhrsSetQuaternion(FusionAhrs *const ahrs, const FusionQuaternion quaternion);

FusionVector FusionAhrsGetGravity(const FusionAhrs *const ahrs);

FusionVector FusionAhrsGetLinearAcceleration(const FusionAhrs *const ahrs);

FusionVector FusionAhrsGetEarthAcceleration(const FusionAhrs *const ahrs);

FusionAhrsInternalStates FusionAhrsGetInternalStates(const FusionAhrs *const ahrs);

FusionAhrsFlags FusionAhrsGetFlags(const FusionAhrs *const ahrs);

void FusionAhrsSetHeading(FusionAhrs *const ahrs, const float heading);

#endif

//------------------------------------------------------------------------------
// End of file
