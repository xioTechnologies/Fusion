/**
 * @file FusionHardIron.h
 * @author Seb Madgwick
 * @brief Run-time estimation and correction of hard-iron offset.
 */

#ifndef FUSION_HARD_IRON_H
#define FUSION_HARD_IRON_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"
#include "FusionProgress.h"
#include "FusionResult.h"
#include <stdbool.h>
#include <stdint.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Include this definition or add as a preprocessor definition to print
 * the heap used.
 */
//#define FUSION_HARD_IRON_PRINT_HEAP_USED

/**
 * @brief Number of magnetometer samples required.
 */
#define FUSION_HARD_IRON_NUMBER_OF_SAMPLES (32)

/**
 * @brief Settings.
 */
typedef struct {
    float sampleRate; // Hz
    float timeout; // seconds
    float intensity; // any calibrated units
} FusionHardIronSettings;

/**
 * @brief Hard-iron structure. All members are private.
 */
typedef struct {
    // Settings
    uint32_t timeout;
    float thresholdSquared;

    // Outputs
    FusionVector magnetometer;
    FusionVector offset;

    // Internal states
    FusionProgressStatus status;
    bool completed;
    uint32_t timer;
    FusionVector samples[FUSION_HARD_IRON_NUMBER_OF_SAMPLES];
    int numberOfSamples;
    float minDistanceSquared;
} FusionHardIron;

//------------------------------------------------------------------------------
// Variable declarations

extern const FusionHardIronSettings fusionHardIronDefaultSettings;

//------------------------------------------------------------------------------
// Function declarations

void FusionHardIronInitialise(FusionHardIron *const hardIron);

void FusionHardIronSetSettings(FusionHardIron *const hardIron, const FusionHardIronSettings *const settings);

FusionResult FusionHardIronUpdate(FusionHardIron *const hardIron, const FusionVector magnetometer);

FusionVector FusionHardIronGetCorrectedMagnetometer(const FusionHardIron *const hardIron);

FusionVector FusionHardIronGetOffset(const FusionHardIron *const hardIron);

void FusionHardIronSetOffset(FusionHardIron *const hardIron, const FusionVector offset);

void FusionHardIronStart(FusionHardIron *const hardIron);

FusionProgress FusionHardIronGetProgress(const FusionHardIron *const hardIron);

FusionResult FusionHardIronComplete(FusionHardIron *const hardIron);

FusionResult FusionHardIronAbort(FusionHardIron *const hardIron);

bool FusionHardIronCompleted(FusionHardIron *const hardIron);

FusionResult FusionHardIronGetSamples(const FusionHardIron *const hardIron, const FusionVector **const samples, int *const numberOfSamples);

FusionResult FusionHardIronSolve(const FusionVector *const samples, const int numberOfSamples, FusionVector *const offset);

#endif

//------------------------------------------------------------------------------
// End of file
