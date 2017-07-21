/**
 * @file FusionTypes.h
 * @author Seb Madgwick
 * @brief Common types and their associated operations.
 *
 * Static inline implementations are used for operations to optimise for
 * increased execution speed.
 */

#ifndef FUSION_TYPES_H
#define FUSION_TYPES_H

//------------------------------------------------------------------------------
// Includes

#include <math.h> // M_PI, sqrtf, atan2f, asinf
#include <stdint.h> // int32_t

//------------------------------------------------------------------------------
// Definitions

/**
 * @brief Three-dimensional spacial vector.
 */
typedef union {
    float array[3];

    struct {
        float x;
        float y;
        float z;
    } axis;
} FusionVector3;

/**
 * @brief Quaternion.  This library uses the conversion of placing the 'w'
 * element as the first element.  Other implementations may place the 'w'
 * element as the last element.
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
 * @brief Rotation matrix in row-major order.
 * @see http://en.wikipedia.org/wiki/Row-major_order
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
} FusionRotationMatrix;

/**
 * @brief Euler angles union.  The Euler angles are in the Aerospace sequence
 * also known as the ZYX sequence.
 */
typedef union {
    float array[3];

    struct {
        float roll;
        float pitch;
        float yaw;
    } angle;
} FusionEulerAngles;

/**
 * @brief Zero-length vector definition.  May be used for initialisation.
 *
 * Example use:
 * @code
 * const FusionVector3 vector3 = FUSION_VECTOR3_ZERO;
 * @endcode
 */
#define FUSION_VECTOR3_ZERO ((FusionVector3){ .array = {0.0f, 0.0f, 0.0f} })

/**
 * @brief Quaternion identity definition to represent an aligned of
 * orientation.  May be used for initialisation.
 *
 * Example use:
 * @code
 * const FusionQuaternion quaternion = FUSION_QUATERNION_IDENTITY;
 * @endcode
 */
#define FUSION_QUATERNION_IDENTITY ((FusionQuaternion){ .array = {1.0f, 0.0f, 0.0f, 0.0f} })

/**
 * @brief Rotation matrix identity definition to represent an aligned of
 * orientation.  May be used for initialisation.
 *
 * Example use:
 * @code
 * const FusionRotationMatrix rotationMatrix = FUSION_ROTATION_MATRIX_IDENTITY;
 * @endcode
 */
#define FUSION_ROTATION_MATRIX_IDENTITY ((FusionRotationMatrix){ .array = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f} })

/**
 * @brief Euler angles zero definition to represent an aligned of orientation.
 * May be used for initialisation.
 *
 * Example use:
 * @code
 * const FusionEulerAngles eulerAngles = FUSION_EULER_ANGLES_ZERO;
 * @endcode
 */
#define FUSION_EULER_ANGLES_ZERO ((FusionEulerAngles){ .array = {0.0f, 0.0f, 0.0f} })

/**
 * @brief Definition of M_PI.  Some compilers may not define this in math.h.
 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Macro for converting degrees to radians.
 */
#define FUSION_DEGREES_TO_RADIANS(degrees) ((float)(degrees) * (M_PI / 180.0f))

/**
 * @brief Macro for converting radians to degrees.
 */
#define FUSION_RADIANS_TO_DEGREES(radians) ((float)(radians) * (180.0f / M_PI))

//------------------------------------------------------------------------------
// Inline functions - Fast inverse square root

/**
 * @brief Calculates the reciprocal of the square root.
 * @see http://en.wikipedia.org/wiki/Fast_inverse_square_root
 * @param x Operand.
 * @return Reciprocal of the square root of x.
 */
static inline __attribute__((always_inline)) float FusionFastInverseSqrt(const float x) {
    float halfx = 0.5f * x;
    float y = x;
    int32_t i = *(int32_t*) & y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*) &i;
    y = y * (1.5f - (halfx * y * y));
    return y;
}

//------------------------------------------------------------------------------
// Inline functions - Vector operations

/**
 * @brief Adds two vectors.
 * @param vectorA First vector of the operation.
 * @param vectorB Second vector of the operation.
 * @return Sum of vectorA and vectorB.
 */
static inline __attribute__((always_inline)) FusionVector3 FusionVectorAdd(const FusionVector3 vectorA, const FusionVector3 vectorB) {
    FusionVector3 resultAdd;
    resultAdd.axis.x = vectorA.axis.x + vectorB.axis.x;
    resultAdd.axis.y = vectorA.axis.y + vectorB.axis.y;
    resultAdd.axis.z = vectorA.axis.z + vectorB.axis.z;
    return resultAdd;
}

/**
 * @brief Subtracts two vectors.
 * @param vectorA First vector of the operation.
 * @param vectorB Second vector of the operation.
 * @return vectorB subtracted from vectorA.
 */
static inline __attribute__((always_inline)) FusionVector3 FusionVectorSubtract(const FusionVector3 vectorA, const FusionVector3 vectorB) {
    FusionVector3 resultSubtract;
    resultSubtract.axis.x = vectorA.axis.x - vectorB.axis.x;
    resultSubtract.axis.y = vectorA.axis.y - vectorB.axis.y;
    resultSubtract.axis.z = vectorA.axis.z - vectorB.axis.z;
    return resultSubtract;
}

/**
 * @brief Multiplies vector by a scalar.
 * @param vector Vector to be multiplied.
 * @param scalar Scalar to be multiplied.
 * @return Vector multiplied by scalar.
 */
static inline __attribute__((always_inline)) FusionVector3 FusionVectorMultiplyScalar(const FusionVector3 vector, const float scalar) {
    FusionVector3 resultMultiplyScalar;
    resultMultiplyScalar.axis.x = vector.axis.x * scalar;
    resultMultiplyScalar.axis.y = vector.axis.y * scalar;
    resultMultiplyScalar.axis.z = vector.axis.z * scalar;
    return resultMultiplyScalar;
}

/**
 * @brief Calculates the cross-product of two vectors.
 * @param vectorA First vector of the operation.
 * @param vectorB Second vector of the operation.
 * @return Cross-product of vectorA and vectorB.
 */
static inline __attribute__((always_inline)) FusionVector3 FusionVectorCrossProduct(const FusionVector3 vectorA, const FusionVector3 vectorB) {
#define A vectorA.axis // define shorthand labels for more readable code
#define B vectorB.axis
    FusionVector3 resultCrossProduct;
    resultCrossProduct.axis.x = A.y * B.z - A.z * B.y;
    resultCrossProduct.axis.y = A.z * B.x - A.x * B.z;
    resultCrossProduct.axis.z = A.x * B.y - A.y * B.x;
    return resultCrossProduct;
#undef A // undefine shorthand labels
#undef B
}

/**
 * @brief Normalises a vector to be of unit magnitude.
 * @param vector Vector to be normalised.
 * @return Normalised vector.
 */
static inline __attribute__((always_inline)) FusionVector3 FusionVectorNormalise(const FusionVector3 vector) {
#define V vector.axis // define shorthand label for more readable code
    const float normReciprocal = 1.0f / sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
    return FusionVectorMultiplyScalar(vector, normReciprocal);
#undef V // undefine shorthand label
}

/**
 * @brief Normalises a vector to be of unit magnitude using the fast inverse
 * square root approximation.
 * @param vector Vector to be normalised.
 * @return Normalised vector.
 */
static inline __attribute__((always_inline)) FusionVector3 FusionVectorFastNormalise(const FusionVector3 vector) {
#define V vector.axis // define shorthand label for more readable code
    const float normReciprocal = FusionFastInverseSqrt(V.x * V.x + V.y * V.y + V.z * V.z);
    return FusionVectorMultiplyScalar(vector, normReciprocal);
#undef V // undefine shorthand label
}

/**
 * @brief Calculates the magnitude of a vector.
 * @param vector Vector to be used in calculation.
 * @return Normalised vector.
 */
static inline __attribute__((always_inline)) float FusionVectorMagnitude(const FusionVector3 vector) {
#define V vector.axis // define shorthand label for more readable code
    return sqrtf(V.x * V.x + V.y * V.y + V.z * V.z);
#undef V // undefine shorthand label
}

//------------------------------------------------------------------------------
// Inline functions - Quaternion operations

/**
 * @brief Adds two quaternions.
 * @param quaternionA First quaternion of the operation.
 * @param quaternionB Second quaternion of the operation.
 * @return Sum of quaternionA and quaternionB.
 */
static inline __attribute__((always_inline)) FusionQuaternion FusionQuaternionAdd(const FusionQuaternion quaternionA, const FusionQuaternion quaternionB) {
    FusionQuaternion result;
    result.element.w = quaternionA.element.w + quaternionB.element.w;
    result.element.x = quaternionA.element.x + quaternionB.element.x;
    result.element.y = quaternionA.element.y + quaternionB.element.y;
    result.element.z = quaternionA.element.z + quaternionB.element.z;
    return result;
}

/**
 * @brief Multiplies two quaternions.
 * @param quaternionA First quaternion of the operation.
 * @param quaternionB Second quaternion of the operation.
 * @return quaternionA multiplied by quaternionB.
 */
static inline __attribute__((always_inline)) FusionQuaternion FusionQuaternionMultiply(const FusionQuaternion quaternionA, const FusionQuaternion quaternionB) {
#define A quaternionA.element // define shorthand labels for more readable code
#define B quaternionB.element
    FusionQuaternion result;
    result.element.w = A.w * B.w - A.x * B.x - A.y * B.y - A.z * B.z;
    result.element.x = A.w * B.x + A.x * B.w + A.y * B.z - A.z * B.y;
    result.element.y = A.w * B.y - A.x * B.z + A.y * B.w + A.z * B.x;
    result.element.z = A.w * B.z + A.x * B.y - A.y * B.x + A.z * B.w;
    return result;
#undef A // undefine shorthand labels
#undef B
}

/**
 * @brief Multiplies quaternion by a vector.  This is a normal quaternion
 * multiplication where the vector is treated a quaternion with a 'w' element
 * value of 0.  The quaternion is post multiplied by the vector.
 * @param quaternion Quaternion to be multiplied.
 * @param vector Vector to be multiplied.
 * @return Quaternion multiplied by vector.
 */
static inline __attribute__((always_inline)) FusionQuaternion FusionQuaternionMultiplyVector(const FusionQuaternion quaternion, const FusionVector3 vector) {
#define Q quaternion.element // define shorthand labels for more readable code
#define V vector.axis
    FusionQuaternion result;
    result.element.w = -Q.x * V.x - Q.y * V.y - Q.z * V.z;
    result.element.x = Q.w * V.x + Q.y * V.z - Q.z * V.y;
    result.element.y = Q.w * V.y - Q.x * V.z + Q.z * V.x;
    result.element.z = Q.w * V.z + Q.x * V.y - Q.y * V.x;
    return result;
#undef Q // undefine shorthand labels
#undef V
}

/**
 * @brief Returns the quaternion conjugate.
 * @param quaternion Quaternion to be conjugated.
 * @return Conjugated quaternion.
 */
static inline __attribute__((always_inline)) FusionQuaternion FusionQuaternionConjugate(const FusionQuaternion quaternion) {
    FusionQuaternion conjugate;
    conjugate.element.w = quaternion.element.w;
    conjugate.element.x = -1.0f * quaternion.element.x;
    conjugate.element.y = -1.0f * quaternion.element.y;
    conjugate.element.z = -1.0f * quaternion.element.z;
    return conjugate;
}

/**
 * @brief Normalises a quaternion to be of unit magnitude.
 * @param quaternion Quaternion to be normalised.
 * @return Normalised quaternion.
 */
static inline __attribute__((always_inline)) FusionQuaternion FusionQuaternionNormalise(const FusionQuaternion quaternion) {
#define Q quaternion.element // define shorthand label for more readable code
    const float normReciprocal = 1.0f / sqrtf(Q.w * Q.w + Q.x * Q.x + Q.y * Q.y + Q.z * Q.z);
    FusionQuaternion normalisedQuaternion;
    normalisedQuaternion.element.w = Q.w * normReciprocal;
    normalisedQuaternion.element.x = Q.x * normReciprocal;
    normalisedQuaternion.element.y = Q.y * normReciprocal;
    normalisedQuaternion.element.z = Q.z * normReciprocal;
    return normalisedQuaternion;
#undef Q // undefine shorthand label
}

/**
 * @brief Normalises a quaternion to be of unit magnitude using the fast inverse
 * square root approximation.
 * @param quaternion Quaternion to be normalised.
 * @return Normalised quaternion.
 */
static inline __attribute__((always_inline)) FusionQuaternion FusionQuaternionFastNormalise(const FusionQuaternion quaternion) {
#define Q quaternion.element // define shorthand label for more readable code
    const float normReciprocal = FusionFastInverseSqrt(Q.w * Q.w + Q.x * Q.x + Q.y * Q.y + Q.z * Q.z);
    FusionQuaternion normalisedQuaternion;
    normalisedQuaternion.element.w = Q.w * normReciprocal;
    normalisedQuaternion.element.x = Q.x * normReciprocal;
    normalisedQuaternion.element.y = Q.y * normReciprocal;
    normalisedQuaternion.element.z = Q.z * normReciprocal;
    return normalisedQuaternion;
#undef Q // undefine shorthand label
}

//------------------------------------------------------------------------------
// Inline functions - Quaternion conversions

/**
 * @brief Converts a quaternion to a rotation matrix.
 *
 * Example use:
 * @code
 * const FusionQuaternion quaternion = FUSION_QUATERNION_IDENTITY;
 * const FusionRotationMatrix rotationMatrix = FusionQuaternionToRotationMatrix(quaternion);
 * @endcode
 *
 * @param quaternion Quaternion to be converted.
 * @return Rotation matrix.
 */
static inline __attribute__((always_inline)) FusionRotationMatrix FusionQuaternionToRotationMatrix(const FusionQuaternion quaternion) {
#define Q quaternion.element // define shorthand label for more readable code
    const float qwqw = Q.w * Q.w; // calculate common terms to avoid repeated operations
    const float qwqx = Q.w * Q.x;
    const float qwqy = Q.w * Q.y;
    const float qwqz = Q.w * Q.z;
    const float qxqy = Q.x * Q.y;
    const float qxqz = Q.x * Q.z;
    const float qyqz = Q.y * Q.z;
    FusionRotationMatrix rotationMatrix;
    rotationMatrix.element.xx = 2.0f * (qwqw - 0.5f + Q.x * Q.x);
    rotationMatrix.element.xy = 2.0f * (qxqy + qwqz);
    rotationMatrix.element.xz = 2.0f * (qxqz - qwqy);
    rotationMatrix.element.yx = 2.0f * (qxqy - qwqz);
    rotationMatrix.element.yy = 2.0f * (qwqw - 0.5f + Q.y * Q.y);
    rotationMatrix.element.yz = 2.0f * (qyqz + qwqx);
    rotationMatrix.element.zx = 2.0f * (qxqz + qwqy);
    rotationMatrix.element.zy = 2.0f * (qyqz - qwqx);
    rotationMatrix.element.zz = 2.0f * (qwqw - 0.5f + Q.z * Q.z);
    return rotationMatrix;
#undef Q // undefine shorthand label
}

/**
 * @brief Converts a quaternion to Euler angles in degrees.
 *
 * Example use:
 * @code
 * const FusionQuaternion quaternion = FUSION_QUATERNION_IDENTITY;
 * const FusionEulerAngles eulerAngles = FusionQuaternionToEulerAngles(quaternion);
 * @endcode
 *
 * @param quaternion Quaternion to be converted.
 * @return Euler angles in degrees.
 */
static inline __attribute__((always_inline)) FusionEulerAngles FusionQuaternionToEulerAngles(const FusionQuaternion quaternion) {
#define Q quaternion.element // define shorthand label for more readable code
    const float qwSquaredMinusHalf = Q.w * Q.w - 0.5f; // calculate common terms to avoid repeated operations
    FusionEulerAngles eulerAngles;
    eulerAngles.angle.roll = FUSION_RADIANS_TO_DEGREES(atan2f(Q.y * Q.z - Q.w * Q.x, qwSquaredMinusHalf + Q.z * Q.z));
    eulerAngles.angle.pitch = FUSION_RADIANS_TO_DEGREES(-1.0f * asinf(2.0f * (Q.x * Q.z + Q.w * Q.y)));
    eulerAngles.angle.yaw = FUSION_RADIANS_TO_DEGREES(atan2f(Q.x * Q.y - Q.w * Q.z, qwSquaredMinusHalf + Q.x * Q.x));
    return eulerAngles;
#undef Q // undefine shorthand label
}

#endif

//------------------------------------------------------------------------------
// End of file
