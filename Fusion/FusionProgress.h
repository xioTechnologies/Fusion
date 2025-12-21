/**
 * @file FusionProgress.h
 * @author Seb Madgwick
 * @brief Progress.
 */

#ifndef FUSION_PROGRESS_H
#define FUSION_PROGRESS_H

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Progress status.
 */
typedef enum {
    FusionProgressStatusNotStarted,
    FusionProgressStatusInProgress,
    FusionProgressStatusComplete,
    FusionProgressStatusFailed,
    FusionProgressStatusAborted,
} FusionProgressStatus;

/**
 * @brief Progress.
 */
typedef struct {
    FusionProgressStatus status;
    unsigned int percentage;
} FusionProgress;

//------------------------------------------------------------------------------
// Function declarations

const char *FusionProgressStatusToString(const FusionProgressStatus status);

#endif

//------------------------------------------------------------------------------
// End of file
