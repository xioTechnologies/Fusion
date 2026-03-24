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
#include "FusionResult.h"
#include <stdbool.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Mode.
 */
typedef enum {
    FusionAhrsModeMagnetic,
    FusionAhrsModeGyroscope,
    FusionAhrsModeExternal,
    FusionAhrsModeAnchored,
} FusionAhrsMode;

/**
 * @brief Settings.
 */
typedef struct {
    FusionAhrsMode mode;
    FusionConvention convention;
    float sampleRate; // Hz
    float gain;
    float gyroscopeRange; // degrees per second
    float accelerationRejection; // degrees
    float magneticRejection; // degrees
    unsigned int recoveryTriggerPeriod; // samples
} FusionAhrsSettings;

/**
 * @brief AHRS structure. All members are private.
 */
typedef struct {
    // Settings
    FusionAhrsSettings settings;
    float samplePeriod;

    // Measurements
    FusionVector accelerometer;
    FusionVector halfGravity;
    FusionQuaternion quaternion;

    // Startup
    bool startup;
    float rampedGain;
    float rampedGainStep;

    // Gyroscope overrange
    bool gyroscopeOverrangeEnabled;
    float gyroscopeOverrangeLimit;
    bool gyroscopeOverrangeRecovery;

    // Acceleration and magnetic rejection
    FusionVector halfAccelerometerFeedback;
    FusionVector halfMagnetometerFeedback;
    bool accelerometerIgnored;
    int accelerationRecoveryTrigger;
    int accelerationRecoveryTimeout;
    bool magnetometerIgnored;
    int magneticRecoveryTrigger;
    int magneticRecoveryTimeout;

    // Anchored heading
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

void FusionAhrsSkipStartup(FusionAhrs *const ahrs);

void FusionAhrsSetSettings(FusionAhrs *const ahrs, const FusionAhrsSettings *const settings);

void FusionAhrsSetSamplePeriod(FusionAhrs *const ahrs, const float samplePeriod);

FusionResult FusionAhrsUpdateMagnetic(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const FusionVector magnetometer);

FusionResult FusionAhrsUpdateGyroscope(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer);

FusionResult FusionAhrsUpdateExternal(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const float heading);

FusionResult FusionAhrsUpdateAnchored(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer);

FusionQuaternion FusionAhrsGetQuaternion(const FusionAhrs *const ahrs);

void FusionAhrsSetQuaternion(FusionAhrs *const ahrs, const FusionQuaternion quaternion);

FusionVector FusionAhrsGetGravity(const FusionAhrs *const ahrs);

FusionVector FusionAhrsGetLinearAcceleration(const FusionAhrs *const ahrs);

FusionVector FusionAhrsGetEarthAcceleration(const FusionAhrs *const ahrs);

FusionAhrsInternalStates FusionAhrsGetInternalStates(const FusionAhrs *const ahrs);

FusionAhrsFlags FusionAhrsGetFlags(const FusionAhrs *const ahrs);

FusionResult FusionAhrsSetHeading(FusionAhrs *const ahrs, const float heading);

FusionResult FusionAhrsSetAnchor(FusionAhrs *const ahrs);

FusionResult FusionAhrsGetAnchorResult(FusionAhrs *const ahrs);

#endif

//------------------------------------------------------------------------------
// End of file
