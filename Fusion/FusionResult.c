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

const char *FusionResultToString(const FusionResult result) {
    switch (result) {
        case FusionResultOk:
            return "Ok";
        case FusionResultTooFewSamples:
            return "Too few samples";
        case FusionResultMallocFailed:
            return "Malloc failed";
    }
    return ""; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
