/**
 * @file FusionResult.h
 * @author Seb Madgwick
 * @brief Result.
 */

#ifndef FUSION_RESULT_H
#define FUSION_RESULT_H

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Result.
 */
typedef enum {
    FusionResultOk,
    FusionResultTooFewSamples,
    FusionResultMallocFailed,
} FusionResult;

//------------------------------------------------------------------------------
// Function declarations

const char *FusionResultToString(const FusionResult result);

#endif

//------------------------------------------------------------------------------
// End of file
