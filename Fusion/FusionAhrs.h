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
#include "FusionProgress.h"
#include "FusionResult.h"
#include <stdbool.h>
#include <stdint.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Heading mode.
 */
typedef enum {
    FusionAhrsHeadingModeMagnetic,
    FusionAhrsHeadingModeRelative,
    FusionAhrsHeadingModeExternal,
    FusionAhrsHeadingModeAnchored,
} FusionAhrsHeadingMode;

/**
 * @brief Settings.
 */
typedef struct {
    float sampleRate; // Hz
    FusionConvention convention;
    FusionAhrsHeadingMode headingMode;
    float gain;
    float gyroscopeRange; // degrees per second (0 = disabled)
    float accelerationRejection; // degrees (0 = disabled)
    float magneticRejection; // degrees (0 = disabled)
    float rejectionTimeout; // seconds
    float anchorCutoff; // Hz
    float anchorDuration; // seconds
} FusionAhrsSettings;

/**
 * @brief AHRS structure. All members are private.
 */
typedef struct {
    // Settings
    float samplePeriod;
    FusionConvention convention;
    FusionAhrsHeadingMode headingMode;
    float inclinationGain;
    float headingGain;
    float startupGainRate;
    bool overrangeEnabled;
    float overrangeThreshold;
    float accelerationRejection;
    float magneticRejection;
    int32_t rejectionTimeout;
    uint32_t anchorDuration;

    // Outputs
    FusionQuaternion quaternion;
    FusionVector accelerometer;

    // Startup
    bool startup;
    float startupGain;

    // Gyroscope overrange
    bool overrangeRecovery;

    // Acceleration rejection
    FusionVector halfAccelerometerResidual;
    int32_t accelerationRecoveryTrigger;
    int32_t accelerationRecoveryThreshold;
    bool accelerometerIgnored;

    // Magnetic rejection
    FusionVector halfMagnetometerResidual;
    int32_t magneticRecoveryTrigger;
    int32_t magneticRecoveryThreshold;
    bool magnetometerIgnored;

    // Anchored heading
    FusionProgressStatus anchorStatus;
    bool anchorCompleted;
    uint32_t anchorNumberOfSamples;
    FusionVector anchorNorth;
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
    bool overrangeRecovery;
    bool accelerationRecovery;
    bool magneticRecovery;
} FusionAhrsFlags;

//------------------------------------------------------------------------------
// Variable declarations

extern const FusionAhrsSettings fusionAhrsDefaultSettings;

//------------------------------------------------------------------------------
// Function declarations

void FusionAhrsInitialise(FusionAhrs *const ahrs);

void FusionAhrsSetSettings(FusionAhrs *const ahrs, const FusionAhrsSettings *const settings);

void FusionAhrsSetSamplePeriod(FusionAhrs *const ahrs, const float samplePeriod);

void FusionAhrsRestart(FusionAhrs *const ahrs);

void FusionAhrsSoftRestart(FusionAhrs *const ahrs);

void FusionAhrsSkipStartup(FusionAhrs *const ahrs);

FusionAhrsHeadingMode FusionAhrsGetHeadingMode(FusionAhrs *const ahrs);

FusionResult FusionAhrsUpdateMagnetic(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer, const FusionVector magnetometer);

FusionResult FusionAhrsUpdateRelative(FusionAhrs *const ahrs, const FusionVector gyroscope, const FusionVector accelerometer);

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

FusionResult FusionAhrsAnchorStart(FusionAhrs *const ahrs);

FusionProgress FusionAhrsAnchorGetProgress(const FusionAhrs *const ahrs);

FusionResult FusionAhrsAnchorComplete(FusionAhrs *const ahrs);

FusionResult FusionAhrsAnchorAbort(FusionAhrs *const ahrs);

bool FusionAhrsAnchorCompleted(FusionAhrs *const ahrs);

const char *FusionAhrsHeadingModeToString(const FusionAhrsHeadingMode headingMode);

#endif

//------------------------------------------------------------------------------
// End of file
