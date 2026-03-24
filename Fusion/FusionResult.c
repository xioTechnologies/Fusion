/**
 * @file FusionResult.c
 * @author Seb Madgwick
 * @brief Result.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionResult.h"

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Returns a string representation of the result.
 * @param result Result.
 * @return String representation of the result.
 */
const char *FusionResultToString(const FusionResult result) {
    switch (result) {
        case FusionResultOk:
            return "Ok";
        case FusionResultInvalidMode:
            return "Invalid mode";
        case FusionResultTooFewSamples:
            return "Too few samples";
        case FusionResultMallocFailed:
            return "Malloc failed";
    }
    return ""; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
