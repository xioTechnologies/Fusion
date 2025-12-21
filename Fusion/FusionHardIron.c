/**
 * @file FusionHardIron.c
 * @author Seb Madgwick
 * @brief Run-time estimation and correction of hard-iron offset.
 */

//------------------------------------------------------------------------------
// Includes

#include <float.h>
#include "FusionHardIron.h"
#include <math.h>
#ifdef FUSION_HARD_IRON_PRINT_HEAP_USED
#include <stdio.h>
#endif
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Threshold to detect sufficient coverage, expressed as a fraction of
 * the ideal sample spacing for points distributed evenly across the surface
 * of a sphere.
 */
#define THRESHOLD (0.7f)

/**
 * @brief Matrix.
 */
typedef struct {
    int rows;
    int cols;
    float array[];
} Matrix;

/**
 * @brief Returns the matrix element.
 * @param row Row.
 * @param col Column.
 * @return Matrix element.
 */
#define ELEMENT(m, row, col) (m->array[row * m->cols + col])

//------------------------------------------------------------------------------
// Function declarations

static inline void AddSample(FusionHardIron *const hardIron, const FusionVector sample);

static Matrix *Empty(const int rows, const int cols);

static Matrix *Copy(const Matrix *const m);

static Matrix *Zeros(const int rows, const int cols);

static Matrix *Identity(const int size);

static Matrix *Multiply(const Matrix *const a, const Matrix *const b);

static Matrix *TransposeMultiply(const Matrix *const a, const Matrix *const b);

static Matrix *MultiplyTranspose(const Matrix *const a, const Matrix *const b);

static Matrix *Inverse(const Matrix *const m);

//------------------------------------------------------------------------------
// Variables

const FusionHardIronSettings fusionHardIronDefaultSettings = {
    .sampleRate = 100.0f,
    .timeout = 30.0f,
    .intensity = 1.0f,
};

#ifdef FUSION_HARD_IRON_PRINT_HEAP_USED
static size_t heapUsed;
#endif

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the hard-iron structure.
 * @param hardIron Hard-iron structure.
 */
void FusionHardIronInitialise(FusionHardIron *const hardIron) {
    FusionHardIronSetSettings(hardIron, &fusionHardIronDefaultSettings);

    hardIron->magnetometer = FUSION_VECTOR_ZERO;
    hardIron->offset = FUSION_VECTOR_ZERO;
    hardIron->status = FusionProgressStatusNotStarted;
    hardIron->completed = false;
    hardIron->timer = 0;
    hardIron->numberOfSamples = 0;
    hardIron->minDistanceSquared = 0.0f;
}

/**
 * @brief Sets the settings.
 * @param hardIron Hard-iron structure.
 * @param settings Settings.
 */
void FusionHardIronSetSettings(FusionHardIron *const hardIron, const FusionHardIronSettings *const settings) {
    hardIron->timeout = (uint32_t) (settings->sampleRate * settings->timeout);

    const float threshold = THRESHOLD * settings->intensity * sqrtf(4.0f * (float) M_PI / FUSION_HARD_IRON_NUMBER_OF_SAMPLES);
    hardIron->thresholdSquared = threshold * threshold;
}

/**
 * @brief Updates the hard-iron algorithm. This function must be called for
 * every magnetometer measurement at the configured sample rate.
 * @param hardIron Hard-iron structure.
 * @param magnetometer Magnetometer in any calibrated units.
 * @return Result.
 */
FusionResult FusionHardIronUpdate(FusionHardIron *const hardIron, const FusionVector magnetometer) {
    hardIron->magnetometer = magnetometer;

    if (hardIron->status != FusionProgressStatusInProgress) {
        return FusionResultOk;
    }

    AddSample(hardIron, magnetometer);

    if (++hardIron->timer >= hardIron->timeout) {
        hardIron->status = FusionProgressStatusFailed;
        return FusionResultTimeout;
    }

    if (hardIron->minDistanceSquared < hardIron->thresholdSquared) {
        return FusionResultOk;
    }

    return FusionHardIronComplete(hardIron);
}

/**
 * @brief Adds a magnetometer sample.
 * @param hardIron Hard-iron structure.
 * @param sample Magnetometer sample.
 */
static inline void AddSample(FusionHardIron *const hardIron, const FusionVector sample) {
    if (hardIron->numberOfSamples < FUSION_HARD_IRON_NUMBER_OF_SAMPLES) {
        hardIron->samples[hardIron->numberOfSamples] = sample;
        hardIron->numberOfSamples++;
        return;
    }

    float minDistanceSquared = FLT_MAX;

    for (int index = 0; index < FUSION_HARD_IRON_NUMBER_OF_SAMPLES; index++) {
        const float distanceSquared = FusionVectorNormSquared(FusionVectorSubtract(hardIron->samples[index], sample));

        if (distanceSquared < minDistanceSquared) {
            minDistanceSquared = distanceSquared;
        }
    }

    int discardIndex = -1;

    for (int candidateIndex = 0; candidateIndex < FUSION_HARD_IRON_NUMBER_OF_SAMPLES; candidateIndex++) {
        for (int neighbourIndex = candidateIndex + 1; neighbourIndex < FUSION_HARD_IRON_NUMBER_OF_SAMPLES; neighbourIndex++) {
            const float distanceSquared = FusionVectorNormSquared(FusionVectorSubtract(hardIron->samples[candidateIndex], hardIron->samples[neighbourIndex]));

            if (distanceSquared < minDistanceSquared) {
                minDistanceSquared = distanceSquared;
                discardIndex = candidateIndex;
            }
        }
    }

    if (discardIndex == -1) {
        return;
    }

    hardIron->samples[discardIndex] = sample;
    hardIron->minDistanceSquared = minDistanceSquared;
}

/**
 * @brief Returns the corrected magnetometer.
 * @param hardIron Hard-iron structure.
 * @return Corrected magnetometer in any calibrated units.
 */
FusionVector FusionHardIronGetCorrectedMagnetometer(const FusionHardIron *const hardIron) {
    return FusionVectorSubtract(hardIron->magnetometer, hardIron->offset);
}

/**
 * @brief Returns the hard-iron offset.
 * @param hardIron Hard-iron structure.
 * @return Hard-iron offset in any calibrated units.
 */
FusionVector FusionHardIronGetOffset(const FusionHardIron *const hardIron) {
    return hardIron->offset;
}

/**
 * @brief Sets the hard-iron offset.
 * @param hardIron Hard-iron structure.
 * @param offset Hard-iron offset in any calibrated units.
 */
void FusionHardIronSetOffset(FusionHardIron *const hardIron, const FusionVector offset) {
    hardIron->offset = offset;
}

/**
 * @brief Starts calibration.
 * @param hardIron Hard-iron structure.
 */
void FusionHardIronStart(FusionHardIron *const hardIron) {
    hardIron->status = FusionProgressStatusInProgress;
    hardIron->completed = false;
    hardIron->timer = 0;
    hardIron->numberOfSamples = 0;
    hardIron->minDistanceSquared = 0.0f;
}

/**
 * @brief Returns the progress.
 * @param hardIron Hard-iron structure.
 * @return Progress.
 */
FusionProgress FusionHardIronGetProgress(const FusionHardIron *const hardIron) {
    const unsigned int percentage = (unsigned int) (100.0f * sqrtf(hardIron->minDistanceSquared / hardIron->thresholdSquared));

    const FusionProgress progress = {
        .status = hardIron->status,
        .percentage = percentage > 100 ? 100 : percentage,
    };
    return progress;
}

/**
 * @brief Completes calibration using the samples collected so far. This
 * function is not normally called by the application because calibration
 * completes automatically once sufficient coverage is detected.
 * @param hardIron Hard-iron structure.
 * @return Result.
 */
FusionResult FusionHardIronComplete(FusionHardIron *const hardIron) {
    if (hardIron->status != FusionProgressStatusInProgress) {
        return FusionResultNotInProgress;
    }

    const FusionVector *samples;
    int numberOfSamples;

    FusionResult result = FusionHardIronGetSamples(hardIron, &samples, &numberOfSamples);

    if (result != FusionResultOk) {
        hardIron->status = FusionProgressStatusFailed;
        return result;
    }

    result = FusionHardIronSolve(samples, numberOfSamples, &hardIron->offset);

    if (result != FusionResultOk) {
        hardIron->status = FusionProgressStatusFailed;
        return result;
    }

    hardIron->status = FusionProgressStatusComplete;
    hardIron->completed = true;
    return FusionResultOk;
}


/**
 * @brief Aborts calibration.
 * @param hardIron Hard-iron structure.
 * @return Result.
 */
FusionResult FusionHardIronAbort(FusionHardIron *const hardIron) {
    if (hardIron->status != FusionProgressStatusInProgress) {
        return FusionResultNotInProgress;
    }

    hardIron->status = FusionProgressStatusAborted;
    return FusionResultOk;
}

/**
 * @brief Returns true if calibration has completed. Calling this function will
 * reset the flag.
 * @param hardIron Hard-iron structure.
 * @return True if calibration has completed.
 */
bool FusionHardIronCompleted(FusionHardIron *const hardIron) {
    const bool completed = hardIron->completed;
    hardIron->completed = false;
    return completed;
}

/**
 * @brief Returns the magnetometer samples.
 * @param hardIron Hard-iron structure.
 * @param samples Magnetometer samples.
 * @param numberOfSamples Number of samples.
 * @return Result.
 */
FusionResult FusionHardIronGetSamples(const FusionHardIron *const hardIron, const FusionVector **const samples, int *const numberOfSamples) {
    if (hardIron->numberOfSamples < FUSION_HARD_IRON_NUMBER_OF_SAMPLES) {
        return FusionResultTooFewSamples;
    }

    *samples = hardIron->samples;
    *numberOfSamples = hardIron->numberOfSamples;
    return FusionResultOk;
}

/**
 * @brief Estimates the hard-iron offset for a set of magnetometer samples by
 * solving the least-squares pseudo-inverse: theta = (A^T * A)^-1 * A^T * y.
 * @param samples Magnetometer samples.
 * @param numberOfSamples Number of samples.
 * @param offset Hard-iron offset in any calibrated units.
 * @return Result.
 */
FusionResult FusionHardIronSolve(const FusionVector *const samples, const int numberOfSamples, FusionVector *const offset) {
    if (numberOfSamples < 4) {
        return FusionResultTooFewSamples;
    }

    FusionResult result = FusionResultMallocFailed;

    Matrix *y = NULL;
    Matrix *a = NULL;
    Matrix *ata = NULL;
    Matrix *inverse = NULL;
    Matrix *pseudo = NULL;
    Matrix *theta = NULL;

    // Create y
    y = Empty(numberOfSamples, 1);
    if (y == NULL) {
        goto cleanup;
    }

    // Create A
    a = Empty(numberOfSamples, 4);
    if (a == NULL) {
        goto cleanup;
    }

    // Populate y and A
    for (int index = 0; index < numberOfSamples; index++) {
        ELEMENT(y, index, 0) = FusionVectorNormSquared(samples[index]);

        ELEMENT(a, index, 0) = samples[index].axis.x;
        ELEMENT(a, index, 1) = samples[index].axis.y;
        ELEMENT(a, index, 2) = samples[index].axis.z;
        ELEMENT(a, index, 3) = 1.0f;
    }

    // ata = A^T * A
    ata = TransposeMultiply(a, a);
    if (ata == NULL) {
        goto cleanup;
    }

    // inverse = (A^T * A)^-1
    inverse = Inverse(ata);
    if (inverse == NULL) {
        goto cleanup;
    }

    // pseudo = (A^T * A)^-1 * A^T
    pseudo = MultiplyTranspose(inverse, a);
    if (pseudo == NULL) {
        goto cleanup;
    }

    // theta = (A^T * A)^-1 * A^T * y
    theta = Multiply(pseudo, y);
    if (theta == NULL) {
        goto cleanup;
    }

    // Extract hard-iron offset from theta
    offset->axis.x = 0.5f * theta->array[0];
    offset->axis.y = 0.5f * theta->array[1];
    offset->axis.z = 0.5f * theta->array[2];

    result = FusionResultOk;

cleanup:
#ifdef FUSION_HARD_IRON_PRINT_HEAP_USED
    printf("Heap used: %zu (%s)\n", heapUsed, FusionResultToString(result));
    heapUsed = 0;
#endif
    free(y);
    free(a);
    free(ata);
    free(inverse);
    free(pseudo);
    free(theta);

    return result;
}

/**
 * @brief Creates an empty matrix.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Empty matrix.
 */
static Matrix *Empty(const int rows, const int cols) {
    const size_t size = sizeof(Matrix) + (rows * cols * sizeof(float));
    Matrix *const m = malloc(size);
    if (m == NULL) {
        return NULL;
    }

#ifdef FUSION_HARD_IRON_PRINT_HEAP_USED
    heapUsed += size;
#endif

    m->rows = rows;
    m->cols = cols;
    return m;
}

/**
 * @brief Copies a matrix.
 * @param m Matrix.
 * @return Copy of the matrix.
 */
static Matrix *Copy(const Matrix *const m) {
    Matrix *const copy = Empty(m->rows, m->cols);
    if (copy == NULL) {
        return NULL;
    }

    memcpy(copy->array, m->array, m->rows * m->cols * sizeof(float));
    return copy;
}

/**
 * @brief Creates a matrix of zeros.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Matrix of zeros.
 */
static Matrix *Zeros(const int rows, const int cols) {
    Matrix *const m = Empty(rows, cols);
    if (m == NULL) {
        return NULL;
    }

    memset(m->array, 0, m->rows * m->cols * sizeof(float));
    return m;
}

/**
 * @brief Creates an identity matrix.
 * @param size Size.
 * @return Identity matrix.
 */
static Matrix *Identity(const int size) {
    Matrix *const m = Zeros(size, size);
    if (m == NULL) {
        return NULL;
    }

    for (int index = 0; index < size; index++) {
        ELEMENT(m, index, index) = 1.0f;
    }
    return m;
}

/**
 * @brief Calculates C = A * B.
 * @param a Matrix A.
 * @param b Matrix B.
 * @return Matrix C.
 */
static Matrix *Multiply(const Matrix *const a, const Matrix *const b) {
    Matrix *const c = Zeros(a->rows, b->cols);
    if (c == NULL) {
        return NULL;
    }

    for (int aRow = 0; aRow < a->rows; aRow++) {
        for (int bCol = 0; bCol < b->cols; bCol++) {
            for (int aCol = 0; aCol < a->cols; aCol++) {
                ELEMENT(c, aRow, bCol) += ELEMENT(a, aRow, aCol) * ELEMENT(b, aCol, bCol);
            }
        }
    }
    return c;
}

/**
 * @brief Calculates C = A^T * B.
 * @param a Matrix A.
 * @param b Matrix B.
 * @return Matrix C.
 */
static Matrix *TransposeMultiply(const Matrix *const a, const Matrix *const b) {
    Matrix *const c = Zeros(a->cols, b->cols);
    if (c == NULL) {
        return NULL;
    }

    for (int aCol = 0; aCol < a->cols; aCol++) {
        for (int bCol = 0; bCol < b->cols; bCol++) {
            for (int aRow = 0; aRow < a->rows; aRow++) {
                ELEMENT(c, aCol, bCol) += ELEMENT(a, aRow, aCol) * ELEMENT(b, aRow, bCol);
            }
        }
    }
    return c;
}

/**
 * @brief Calculates C = A * B^T.
 * @param a Matrix A.
 * @param b Matrix B.
 * @return Matrix C.
 */
static Matrix *MultiplyTranspose(const Matrix *const a, const Matrix *const b) {
    Matrix *const c = Zeros(a->rows, b->rows);
    if (c == NULL) {
        return NULL;
    }

    for (int aRow = 0; aRow < a->rows; aRow++) {
        for (int bRow = 0; bRow < b->rows; bRow++) {
            for (int aCol = 0; aCol < a->cols; aCol++) {
                ELEMENT(c, aRow, bRow) += ELEMENT(a, aRow, aCol) * ELEMENT(b, bRow, aCol);
            }
        }
    }
    return c;
}

/**
 * @brief Calculates M^-1 using Gauss–Jordan elimination. M must be a symmetric
 * positive definite matrix, which is true for M = A^T * A.
 * @param m Matrix.
 * @return Matrix inverse.
 */
static Matrix *Inverse(const Matrix *const m) {
    const int size = m->rows;

    Matrix *working = NULL;
    Matrix *inverse = NULL;

    working = Copy(m);
    if (working == NULL) {
        goto cleanup;
    }

    inverse = Identity(size);
    if (inverse == NULL) {
        goto cleanup;
    }

    for (int pivotRow = 0; pivotRow < size; pivotRow++) {
        const float pivotValue = ELEMENT(working, pivotRow, pivotRow);

        for (int pivotCol = 0; pivotCol < size; pivotCol++) {
            ELEMENT(working, pivotRow, pivotCol) /= pivotValue;
            ELEMENT(inverse, pivotRow, pivotCol) /= pivotValue;
        }

        for (int targetRow = 0; targetRow < size; targetRow++) {
            if (targetRow == pivotRow) {
                continue;
            }

            const float factor = ELEMENT(working, targetRow, pivotRow);

            for (int targetCol = 0; targetCol < size; targetCol++) {
                ELEMENT(working, targetRow, targetCol) -= factor * ELEMENT(working, pivotRow, targetCol);
                ELEMENT(inverse, targetRow, targetCol) -= factor * ELEMENT(inverse, pivotRow, targetCol);
            }
        }
    }

cleanup:
    free(working);

    return inverse;
}

//------------------------------------------------------------------------------
// End of file
