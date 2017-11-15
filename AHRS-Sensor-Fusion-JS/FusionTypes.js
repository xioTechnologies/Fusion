const FusionVector3 = '{"axis":{"x":0,"y":0,"z":0}}';
const FusionQuaternion = '{"element":{"w":0,"x":0,"y":0,"z":0}}';
const FusionRotationMatrix = '{"element":{"xx":0,"xy":0,"xz":0,"yx":0,"yy":0,"yz":0,"zx":0,"zy":0,"zz":0}}';
const FusionEulerAngles = '{"angle":{"roll":0,"pitch":0,"yaw":0}}';
const FUSION_VECTOR3_ZERO = '{"axis":{"x":0,"y":0,"z":0}}';
const FUSION_QUATERNION_IDENTITY = '{"element":{"w":0,"x":0,"y":0,"z":0}}';
const FUSION_ROTATION_MATRIX_IDENTITY = '{"element":{"xx":0,"xy":0,"xz":0,"yx":0,"yy":0,"yz":0,"zx":0,"zy":0,"zz":0}}';
const FUSION_EULER_ANGLES_ZERO = '{"roll":0,"pitch":0,"yaw":0}';
const M_PI = 3.14159265358979323846;

function FUSION_DEGREES_TO_RADIANS(degrees)
{
  return degrees * (M_PI / 180.0);
}

function FUSION_RADIANS_TO_DEGREES(radians)
{
  return radians * (180.0 / M_PI);
}

function FusionFastInverseSqrt(x)
{
  var buf = new ArrayBuffer(4),
      f32=new Float32Array(buf),
      u32=new Uint32Array(buf),
      x2 = 0.5 * (f32[0] = x);

  u32[0] = (0x5f3759df - (u32[0] >> 1));
  var y = f32[0];
  y  = y * ( 1.5 - ( x2 * y * y ) );
  return y;
}

function FusionVectorAdd(vectorA, vectorB)
{
  var result = clone(FusionVector3);
  result.axis.x = vectorA.axis.x + vectorB.axis.x;
  result.axis.y = vectorA.axis.y + vectorB.axis.y;
  result.axis.z = vectorA.axis.z + vectorB.axis.z;
  return result;
}

function FusionVectorSubtract(vectorA,vectorB)
{
  var result = clone(FusionVector3);
  result.axis.x = vectorA.axis.x - vectorB.axis.x;
  result.axis.y = vectorA.axis.y - vectorB.axis.y;
  result.axis.z = vectorA.axis.z - vectorB.axis.z;
  return result;
}

function FusionVectorMultiplyScalar(vector, scalar)
{
  var result = clone(FusionVector3);
  result.axis.x = vector.axis.x * scalar;
  result.axis.y = vector.axis.y * scalar;
  result.axis.z = vector.axis.z * scalar;
  return result;
}

function FusionVectorCrossProduct(vectorA,vectorB)
{
  var A = vectorA.axis;
  var B = vectorB.axis;
  var result = clone(FusionVector3);
  result.axis.x = A.y * B.z - A.z * B.y;
  result.axis.y = A.z * B.x - A.x * B.z;
  result.axis.z = A.x * B.y - A.y * B.x;
  return result;
}

function FusionVectorNormalise(vector)
{
  var V = vector.axis;
  var normReciprocal = 1.0 / Math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
  return FusionVectorMultiplyScalar(vector, normReciprocal);
}

function FusionVectorFastNormalise(vector)
{
  var V = vector.axis;
  var normReciprocal = FusionFastInverseSqrt(V.x * V.x + V.y * V.y + V.z * V.z);
  return FusionVectorMultiplyScalar(vector, normReciprocal);
}

function FusionVectorMagnitude(vector)
{
  var V = vector.axis;
  return Math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
}

function FusionQuaternionAdd(quaternionA, quaternionB)
{
  var result = clone(FusionQuaternion);
  result.element.w = quaternionA.element.w + quaternionB.element.w;
  result.element.x = quaternionA.element.x + quaternionB.element.x;
  result.element.y = quaternionA.element.y + quaternionB.element.y;
  result.element.z = quaternionA.element.z + quaternionB.element.z;
  return result;
}

function FusionQuaternionMultiply(quaternionA, quaternionB)
{
  var A = quaternionA.element;
  var B = quaternionB.element;
  var result = clone(FusionQuaternion);
  result.element.w = A.w * B.w - A.x * B.x - A.y * B.y - A.z * B.z;
  result.element.x = A.w * B.x + A.x * B.w + A.y * B.z - A.z * B.y;
  result.element.y = A.w * B.y - A.x * B.z + A.y * B.w + A.z * B.x;
  result.element.z = A.w * B.z + A.x * B.y - A.y * B.x + A.z * B.w;
  return result;
}

function FusionQuaternionMultiplyVector(quaternion, vector)
{
  var Q = quaternion.element;
  var V = vector.axis;
  var result = clone(FusionQuaternion);
  result.element.w = -Q.x * V.x - Q.y * V.y - Q.z * V.z;
  result.element.x = Q.w * V.x + Q.y * V.z - Q.z * V.y;
  result.element.y = Q.w * V.y - Q.x * V.z + Q.z * V.x;
  result.element.z = Q.w * V.z + Q.x * V.y - Q.y * V.x;
  return result;
}

function FusionQuaternionConjugate(quaternion)
{
  var conjugate = clone(FusionQuaternion);
  conjugate.element.w = quaternion.element.w;
  conjugate.element.x = -1.0 * quaternion.element.x;
  conjugate.element.y = -1.0 * quaternion.element.y;
  conjugate.element.z = -1.0 * quaternion.element.z;
  return conjugate;
}

function FusionQuaternionNormalise(quaternion)
{
  var Q = quaternion.element;
  var normReciprocal = 1.0 / Math.sqrt(Q.w * Q.w + Q.x * Q.x + Q.y * Q.y + Q.z * Q.z);
  var normalisedQuaternion = clone(FusionQuaternion);
  normalisedQuaternion.element.w = Q.w * normReciprocal;
  normalisedQuaternion.element.x = Q.x * normReciprocal;
  normalisedQuaternion.element.y = Q.y * normReciprocal;
  normalisedQuaternion.element.z = Q.z * normReciprocal;
  return normalisedQuaternion;
}

function FusionQuaternionFastNormalise(quaternion)
{
  var Q = quaternion.element;
  var normReciprocal = FusionFastInverseSqrt(Q.w * Q.w + Q.x * Q.x + Q.y * Q.y + Q.z * Q.z);
  var normalisedQuaternion = clone(FusionQuaternion);
  normalisedQuaternion.element.w = Q.w * normReciprocal;
  normalisedQuaternion.element.x = Q.x * normReciprocal;
  normalisedQuaternion.element.y = Q.y * normReciprocal;
  normalisedQuaternion.element.z = Q.z * normReciprocal;
  return normalisedQuaternion;
}

function FusionQuaternionToRotationMatrix(quaternion)
{
  var Q = quaternion.element;
  var qwqw = Q.w * Q.w;
  var qwqx = Q.w * Q.x;
  var qwqy = Q.w * Q.y;
  var qwqz = Q.w * Q.z;
  var qxqy = Q.x * Q.y;
  var qxqz = Q.x * Q.z;
  var qyqz = Q.y * Q.z;
  var rotationMatrix = clone(FusionRotationMatrix);
  rotationMatrix.element.xx = 2.0 * (qwqw - 0.5 + Q.x * Q.x);
  rotationMatrix.element.xy = 2.0 * (qxqy + qwqz);
  rotationMatrix.element.xz = 2.0 * (qxqz - qwqy);
  rotationMatrix.element.yx = 2.0 * (qxqy - qwqz);
  rotationMatrix.element.yy = 2.0 * (qwqw - 0.5 + Q.y * Q.y);
  rotationMatrix.element.yz = 2.0 * (qyqz + qwqx);
  rotationMatrix.element.zx = 2.0 * (qxqz + qwqy);
  rotationMatrix.element.zy = 2.0 * (qyqz - qwqx);
  rotationMatrix.element.zz = 2.0 * (qwqw - 0.5 + Q.z * Q.z);
  return rotationMatrix;
}

function FusionQuaternionToEulerAngles(quaternion)
{
  var Q = quaternion.element;
  var qwSquaredMinusHalf = Q.w * Q.w - 0.5;
  var eulerAngles = clone(FusionEulerAngles);
  eulerAngles.angle.roll = FUSION_RADIANS_TO_DEGREES(Math.atan2(Q.y * Q.z - Q.w * Q.x, qwSquaredMinusHalf + Q.z * Q.z));
  eulerAngles.angle.pitch = FUSION_RADIANS_TO_DEGREES(-1.0 * Math.asin(2.0 * (Q.x * Q.z + Q.w * Q.y)));
  eulerAngles.angle.yaw = FUSION_RADIANS_TO_DEGREES(Math.atan2(Q.x * Q.y - Q.w * Q.z, qwSquaredMinusHalf + Q.x * Q.x));
  return eulerAngles;
}

function clone (object)
{
  return JSON.parse(object);
}
