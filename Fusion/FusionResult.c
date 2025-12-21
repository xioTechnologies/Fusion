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
        case FusionResultNotInProgress:
            return "Not in progress";
        case FusionResultTooFewSamples:
            return "Too few samples";
        case FusionResultTimeout:
            return "Timeout";
        case FusionResultMallocFailed:
            return "Malloc failed";
        case FusionResultSingularMatrix:
            return "Singular matrix";
    }
    return ""; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
