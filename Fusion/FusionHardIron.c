/**
 * @file FusionHardIron.c
 * @author Seb Madgwick
 * @brief Run-time estimation and compensation of hard-iron offset.
 */

//------------------------------------------------------------------------------
// Includes

#include "FusionHardIron.h"
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Definitions

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
};

//------------------------------------------------------------------------------
// Functions

/**
 * @brief Initialises the hard-iron structure.
 * @param hardIron Hard-iron structure.
 */
void FusionHardIronInitialise(FusionHardIron *const hardIron) {
    FusionHardIronSetSettings(hardIron, &fusionHardIronDefaultSettings);
    hardIron->offset = FUSION_VECTOR_ZERO;
}

/**
 * @brief Sets the settings.
 * @param hardIron Hard-iron structure.
 * @param settings Settings.
 */
void FusionHardIronSetSettings(FusionHardIron *const hardIron, const FusionHardIronSettings *const settings) {
    hardIron->timeout = (unsigned int) (settings->sampleRate * settings->timeout);
}

/**
 * @brief Updates the hard-iron algorithm and returns the hard-iron-corrected
 * magnetometer. This function must be called for every gyroscope sample at the
 * configured sample rate.
 * @param hardIron Hard-iron structure.
 * @param gyroscope Gyroscope in degrees per second.
 * @param magnetometer Magnetometer in any calibrated units.
 * @return Hard-iron-corrected magnetometer.
 */
FusionVector FusionHardIronUpdate(FusionHardIron *const hardIron, const FusionVector gyroscope, FusionVector magnetometer) {
    magnetometer = FusionVectorSubtract(magnetometer, hardIron->offset);
    return magnetometer;
}

/**
 * @brief Estimates the hard-iron offset for a set of magnetometer samples by
 * solving the least-squares pseudo-inverse: theta = (A^T * A)^-1 * A^T * y.
 * @param samples Magnetometer samples.
 * @param numberOfSamples Number of samples.
 * @param offset Hard-iron offset.
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

    // inverse = (A * A^T)^-1
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
    Matrix *const m = malloc(sizeof(Matrix) + (rows * cols * sizeof(float)));
    if (m == NULL) {
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    return m;
}

/**
 * @brief Copies matrix.
 * @param m Matrix.
 * @return Copy of matrix.
 */
static Matrix *Copy(const Matrix *const m) {
    const size_t size = sizeof(Matrix) + (m->rows * m->cols * sizeof(float));

    Matrix *const copy = malloc(size);
    if (copy == NULL) {
        return NULL;
    }

    memcpy(copy, m, size);
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

    memset(m->array, 0, rows * cols * sizeof(float));
    return m;
}

/**
 * @brief Creates an identity matrix.
 * @param size Size.
 * @return Identity matrix.
 */
static Matrix *Identity(const int size) {
    Matrix *const m = Empty(size, size);
    if (m == NULL) {
        return NULL;
    }

    memset(m->array, 0, size * size * sizeof(float));

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

    Matrix *const working = Copy(m);
    Matrix *const inverse = Identity(size);

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

    free(working);
    return inverse;
}

//------------------------------------------------------------------------------
// End of file
