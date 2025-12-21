/**
 * @file FusionProgress.c
 * @author Seb Madgwick
 * @brief Progress.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionProgress.h"

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Returns a string representation of the progress status.
 * @param status Progress status.
 * @return String representation of the progress status.
 */
const char *FusionProgressStatusToString(const FusionProgressStatus status) {
    switch (status) {
        case FusionProgressStatusNotStarted:
            return "Not started";
        case FusionProgressStatusInProgress:
            return "In progress";
        case FusionProgressStatusComplete:
            return "Complete";
        case FusionProgressStatusFailed:
            return "Failed";
        case FusionProgressStatusAborted:
            return "Aborted";
    }
    return ""; // avoid compiler warning
}

//------------------------------------------------------------------------------
// End of file
