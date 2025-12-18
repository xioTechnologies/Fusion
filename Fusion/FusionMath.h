/**
 * @file FusionMath.h
 * @author Seb Madgwick
 * @brief Math library.
 */

#ifndef FUSION_MATH_H
#define FUSION_MATH_H

//------------------------------------------------------------------------------
// Includes

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief 3D vector.
 */
typedef union {
    float array[3];

    struct {
        float x;
        float y;
        float z;
    } axis;
} FusionVector;

/**
 * @brief Quaternion.
 */
typedef union {
    float array[4];

    struct {
        float w;
        float x;
        float y;
        float z;
    } element;
} FusionQuaternion;

/**
 * @brief 3x3 matrix in row-major order.
 * See http://en.wikipedia.org/wiki/Row-major_order
 */
typedef union {
    float array[9];

    struct {
        float xx;
        float xy;
        float xz;
        float yx;
        float yy;
        float yz;
        float zx;
        float zy;
        float zz;
    } element;
} FusionMatrix;

/**
 * @brief ZYX Euler angles in degrees. Roll, pitch, and yaw are rotations
 * around X, Y, and Z respectively.
 */
typedef union {
    float array[3];

    struct {
        float roll;
        float pitch;
        float yaw;
    } angle;
} FusionEuler;

/**
 * @brief Vector of zeros.
 */
#define FUSION_VECTOR_ZERO ((FusionVector){ .array = {0.0f, 0.0f, 0.0f} })

/**
 * @brief Vector of ones.
 */
#define FUSION_VECTOR_ONES ((FusionVector){ .array = {1.0f, 1.0f, 1.0f} })

/**
 * @brief Identity quaternion.
 */
#define FUSION_QUATERNION_IDENTITY ((FusionQuaternion){ .array = {1.0f, 0.0f, 0.0f, 0.0f} })

/**
 * @brief Identity matrix.
 */
#define FUSION_MATRIX_IDENTITY ((FusionMatrix){ .array = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}} })

/**
 * @brief Euler angles of zero.
 */
#define FUSION_EULER_ZERO ((FusionEuler){ .array = {0.0f, 0.0f, 0.0f} })

/**
 * @brief Pi. May not be defined in math.h.
 */
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

/**
 * @brief Include this definition or add as a preprocessor definition to use
 * normal square root operations.
 */
//#define FUSION_USE_NORMAL_SQRT

//------------------------------------------------------------------------------
// Inline functions - Degrees and radians conversion

/**
 * @brief Converts degrees to radians.
 * @param degrees Degrees.
 * @return Radians.
 */
static inline float FusionDegreesToRadians(const float degrees) {
    return degrees * ((float) M_PI / 180.0f);
}

/**
 * @brief Converts radians to degrees.
 * @param radians Radians.
 * @return Degrees.
 */
static inline float FusionRadiansToDegrees(const float radians) {
    return radians * (180.0f / (float) M_PI);
}

//------------------------------------------------------------------------------
// Inline functions - Arc sine

/**
 * @brief Returns the arc sine of a value. Out of range values are clamped to
 * avoid a NaN result.
 * @param value Value.
 * @return Arc sine of the value.
 */
static inline float FusionArcSin(const float value) {
    if (value <= -1.0f) {
        return (float) M_PI / -2.0f;
    }
    if (value >= 1.0f) {
        return (float) M_PI / 2.0f;
    }
    return asinf(value);
}

//------------------------------------------------------------------------------
// Inline functions - Fast inverse square root

#ifndef FUSION_USE_NORMAL_SQRT

/**
 * @brief Calculates the reciprocal of the square root.
 * See https://pizer.wordpress.com/2008/10/12/fast-inverse-square-root/
 * @param x Operand.
 * @return Reciprocal of the square root of x.
 */
static inline float FusionFastInverseSqrt(const float x) {
    typedef union {
        float f;
        int32_t i;
    } Union32;

    Union32 union32 = {.f = x};
    union32.i = 0x5F1F1412 - (union32.i >> 1);
    return union32.f * (1.69000231f - 0.714158168f * x * union32.f * union32.f);
}

#endif

//------------------------------------------------------------------------------
// Inline functions - Vector operations

/**
 * @brief Returns true if the vector is zero.
 * @param v Vector.
 * @return True if the vector is zero.
 */
static inline bool FusionVectorIsZero(const FusionVector v) {
    return (v.axis.x == 0.0f) && (v.axis.y == 0.0f) && (v.axis.z == 0.0f);
}

/**
 * @brief Returns the sum of two vectors.
 * @param a Vector a.
 * @param b Vector b.
 * @return Sum of two vectors.
 */
static inline FusionVector FusionVectorAdd(const FusionVector a, const FusionVector b) {
    const FusionVector result = {
        .axis = {
            .x = a.axis.x + b.axis.x,
            .y = a.axis.y + b.axis.y,
            .z = a.axis.z + b.axis.z,
        }
    };
    return result;
}

/**
 * @brief Returns the subtraction of two vectors: a - b.
 * @param a Vector a.
 * @param b Vector b.
 * @return Subtraction of two vectors.
 */
static inline FusionVector FusionVectorSubtract(const FusionVector a, const FusionVector b) {
    const FusionVector result = {
        .axis = {
            .x = a.axis.x - b.axis.x,
            .y = a.axis.y - b.axis.y,
            .z = a.axis.z - b.axis.z,
        }
    };
    return result;
}

/**
 * @brief Returns a scaled vector.
 * @param v Vector.
 * @param s Scalar.
 * @return Scaled vector.
 */
static inline FusionVector FusionVectorScale(const FusionVector v, const float s) {
    const FusionVector result = {
        .axis = {
            .x = v.axis.x * s,
            .y = v.axis.y * s,
            .z = v.axis.z * s,
        }
    };
    return result;
}

/**
 * @brief Returns the sum of the elements.
 * @param v Vector.
 * @return Sum of the elements.
 */
static inline float FusionVectorSum(const FusionVector v) {
    return v.axis.x + v.axis.y + v.axis.z;
}

/**
 * @brief Returns the Hadamard (element-wise) product.
 * @param a Vector a.
 * @param b Vector b.
 * @return Hadamard (element-wise) product.
 */
static inline FusionVector FusionVectorHadamard(const FusionVector a, const FusionVector b) {
    const FusionVector result = {
        .axis = {
            .x = a.axis.x * b.axis.x,
            .y = a.axis.y * b.axis.y,
            .z = a.axis.z * b.axis.z,
        }
    };
    return result;
}

/**
 * @brief Returns the cross product: a cross b.
 * @param a Vector a.
 * @param b Vector b.
 * @return Cross product.
 */
static inline FusionVector FusionVectorCross(const FusionVector a, const FusionVector b) {
    const FusionVector result = {
        .axis = {
            .x = a.axis.y * b.axis.z - a.axis.z * b.axis.y,
            .y = a.axis.z * b.axis.x - a.axis.x * b.axis.z,
            .z = a.axis.x * b.axis.y - a.axis.y * b.axis.x,
        }
    };
    return result;
}

/**
 * @brief Returns the dot product.
 * @param a Vector a.
 * @param b Vector b.
 * @return Dot product.
 */
static inline float FusionVectorDot(const FusionVector a, const FusionVector b) {
    return FusionVectorSum(FusionVectorHadamard(a, b));
}

/**
 * @brief Returns the vector magnitude squared.
 * @param v Vector.
 * @return Vector magnitude squared.
 */
static inline float FusionVectorNormSquared(const FusionVector v) {
    return FusionVectorSum(FusionVectorHadamard(v, v));
}

/**
 * @brief Returns the vector magnitude.
 * @param v Vector.
 * @return Vector magnitude.
 */
static inline float FusionVectorNorm(const FusionVector v) {
    return sqrtf(FusionVectorNormSquared(v));
}

/**
 * @brief Returns the normalised vector.
 * @param v Vector.
 * @return Normalised vector.
 */
static inline FusionVector FusionVectorNormalise(const FusionVector v) {
#ifdef FUSION_USE_NORMAL_SQRT
    return FusionVectorScale(v, 1.0f / FusionVectorNorm(v));
#else
    return FusionVectorScale(v, FusionFastInverseSqrt(FusionVectorNormSquared(v)));
#endif
}

//------------------------------------------------------------------------------
// Inline functions - Quaternion operations

/**
 * @brief Returns the sum of two quaternions.
 * @param a Quaternion a.
 * @param b Quaternion b.
 * @return Sum of two quaternions.
 */
static inline FusionQuaternion FusionQuaternionAdd(const FusionQuaternion a, const FusionQuaternion b) {
    const FusionQuaternion result = {
        .element = {
            .w = a.element.w + b.element.w,
            .x = a.element.x + b.element.x,
            .y = a.element.y + b.element.y,
            .z = a.element.z + b.element.z,
        }
    };
    return result;
}

/**
 * @brief Returns a scaled quaternion.
 * @param q Quaternion.
 * @param s Scalar.
 * @return Scaled quaternion.
 */
static inline FusionQuaternion FusionQuaternionScale(const FusionQuaternion q, const float s) {
    const FusionQuaternion result = {
        .element = {
            .w = q.element.w * s,
            .x = q.element.x * s,
            .y = q.element.y * s,
            .z = q.element.z * s,
        }
    };
    return result;
}

/**
 * @brief Returns the sum of the elements.
 * @param q Quaternion.
 * @return Sum of the elements.
 */
static inline float FusionQuaternionSum(const FusionQuaternion q) {
    return q.element.w + q.element.x + q.element.y + q.element.z;
}

/**
 * @brief Returns the Hadamard (element-wise) product.
 * @param a Quaternion a.
 * @param b Quaternion b.
 * @return Hadamard (element-wise) product.
 */
static inline FusionQuaternion FusionQuaternionHadamard(const FusionQuaternion a, const FusionQuaternion b) {
    const FusionQuaternion result = {
        .element = {
            .w = a.element.w * b.element.w,
            .x = a.element.x * b.element.x,
            .y = a.element.y * b.element.y,
            .z = a.element.z * b.element.z,
        }
    };
    return result;
}

/**
 * @brief Returns the quaternion product: a * b.
 * @param a Quaternion a.
 * @param b Quaternion b.
 * @return Quaternion product.
 */
static inline FusionQuaternion FusionQuaternionProduct(const FusionQuaternion a, const FusionQuaternion b) {
#define A a.element
#define B b.element
    const FusionQuaternion result = {
        .element = {
            .w = A.w * B.w - A.x * B.x - A.y * B.y - A.z * B.z,
            .x = A.w * B.x + A.x * B.w + A.y * B.z - A.z * B.y,
            .y = A.w * B.y - A.x * B.z + A.y * B.w + A.z * B.x,
            .z = A.w * B.z + A.x * B.y - A.y * B.x + A.z * B.w,
        }
    };
#undef A
#undef B
    return result;
}

/**
 * @brief Returns the quaternion-vector product: q * v. The vector is treated
 * as a quaternion with w = 0.
 * @param q Quaternion.
 * @param v Vector.
 * @return Quaternion-vector product.
 */
static inline FusionQuaternion FusionQuaternionVectorProduct(const FusionQuaternion q, const FusionVector v) {
#define Q q.element
#define V v.axis
    const FusionQuaternion result = {
        .element = {
            .w = -Q.x * V.x - Q.y * V.y - Q.z * V.z,
            .x = Q.w * V.x + Q.y * V.z - Q.z * V.y,
            .y = Q.w * V.y - Q.x * V.z + Q.z * V.x,
            .z = Q.w * V.z + Q.x * V.y - Q.y * V.x,
        }
    };
#undef Q
#undef V
    return result;
}

/**
 * @brief Returns the quaternion norm squared.
 * @param q Quaternion.
 * @return Quaternion norm squared.
 */
static inline float FusionQuaternionNormSquared(const FusionQuaternion q) {
    return FusionQuaternionSum(FusionQuaternionHadamard(q, q));
}

/**
 * @brief Returns the quaternion norm.
 * @param q Quaternion.
 * @return Quaternion norm.
 */
static inline float FusionQuaternionNorm(const FusionQuaternion q) {
    return sqrtf(FusionQuaternionNormSquared(q));
}

/**
 * @brief Returns the normalised quaternion.
 * @param q Quaternion.
 * @return Normalised quaternion.
 */
static inline FusionQuaternion FusionQuaternionNormalise(const FusionQuaternion q) {
#ifdef FUSION_USE_NORMAL_SQRT
    return FusionQuaternionScale(q, 1.0f / FusionQuaternionNorm(q));
#else
    return FusionQuaternionScale(q, FusionFastInverseSqrt(FusionQuaternionNormSquared(q)));
#endif
}

//------------------------------------------------------------------------------
// Inline functions - Matrix operations

/**
 * @brief Returns the multiplication of a matrix and a vector: M * v.
 * @param m Matrix.
 * @param v Vector.
 * @return Multiplication of a matrix and a vector.
 */
static inline FusionVector FusionMatrixMultiply(const FusionMatrix m, const FusionVector v) {
#define M m.element
#define V v.axis
    const FusionVector result = {
        .axis = {
            .x = M.xx * V.x + M.xy * V.y + M.xz * V.z,
            .y = M.yx * V.x + M.yy * V.y + M.yz * V.z,
            .z = M.zx * V.x + M.zy * V.y + M.zz * V.z,
        }
    };
#undef M
#undef V
    return result;
}

//------------------------------------------------------------------------------
// Inline functions - Conversions

/**
 * @brief Converts a quaternion to a rotation matrix.
 *
 * Quaternions and Rotation Sequences by Jack B. Kuipers, ISBN 0-691-10298-8,
 * Page 168. The matrix is the transpose of that shown in the book.
 *
 * @param q Quaternion.
 * @return Rotation matrix.
 */
static inline FusionMatrix FusionQuaternionToMatrix(const FusionQuaternion q) {
#define Q q.element
    const float twoQw = 2.0f * Q.w;
    const float twoQx = 2.0f * Q.x;
    const float twoQy = 2.0f * Q.y;
    const float twoQz = 2.0f * Q.z;
    const FusionMatrix matrix = {
        .element = {
            .xx = twoQw * Q.w - 1.0f + twoQx * Q.x,
            .xy = twoQx * Q.y - twoQw * Q.z,
            .xz = twoQx * Q.z + twoQw * Q.y,
            .yx = twoQx * Q.y + twoQw * Q.z,
            .yy = twoQw * Q.w - 1.0f + twoQy * Q.y,
            .yz = twoQy * Q.z - twoQw * Q.x,
            .zx = twoQx * Q.z - twoQw * Q.y,
            .zy = twoQy * Q.z + twoQw * Q.x,
            .zz = twoQw * Q.w - 1.0f + twoQz * Q.z,
        }
    };
#undef Q
    return matrix;
}

/**
 * @brief Converts a quaternion to Euler angles.
 *
 * Quaternions and Rotation Sequences by Jack B. Kuipers, ISBN 0-691-10298-8,
 * Page 168.
 *
 * @param q Quaternion.
 * @return Euler angles.
 */
static inline FusionEuler FusionQuaternionToEuler(const FusionQuaternion q) {
#define Q q.element
    const FusionEuler euler = {
        .angle = {
            .roll = FusionRadiansToDegrees(atan2f(Q.y * Q.z + Q.w * Q.x, Q.w * Q.w + Q.z * Q.z - 0.5f)),
            .pitch = FusionRadiansToDegrees(FusionArcSin(2.0f * (Q.w * Q.y - Q.x * Q.z))),
            .yaw = FusionRadiansToDegrees(atan2f(Q.x * Q.y + Q.w * Q.z, Q.w * Q.w + Q.x * Q.x - 0.5f)),
        }
    };
#undef Q
    return euler;
}

#endif

//------------------------------------------------------------------------------
// End of file
