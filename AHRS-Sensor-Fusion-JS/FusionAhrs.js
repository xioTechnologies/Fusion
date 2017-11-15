const FusionAhrs = '{"gain":0,"minMagneticFieldSquared":0,"maxMagneticFieldSquared":0,"quaternion":{"element":{"w":0,"x":0,"y":0,"z":0}},"linearAcceleration":{"axis":{"x":0,"y":0,"z":0}},"rampedGain":0}';
const INITIAL_GAIN = 10.0;
const INITIALISATION_PERIOD = 3.0;

function FusionAhrsInitialise(fusionAhrs, gain, minMagneticField, maxMagneticField)
{
    fusionAhrs.gain = gain;
    fusionAhrs.minMagneticFieldSquared = minMagneticField * minMagneticField;
    fusionAhrs.maxMagneticFieldSquared = maxMagneticField * maxMagneticField;
    fusionAhrs.quaternion = clone(FUSION_QUATERNION_IDENTITY);
    fusionAhrs.linearAcceleration = clone(FUSION_VECTOR3_ZERO);
    fusionAhrs.rampedGain = clone(INITIAL_GAIN);
    fusionAhrs.quaternion.element.w = 1.0;
    fusionAhrs.quaternion.element.x = 0.0;
    fusionAhrs.quaternion.element.y = 0.0;
    fusionAhrs.quaternion.element.z = 0.0;
}

function FusionAhrsUpdate(fusionAhrs, gyroscope, accelerometer, magnetometer, samplePeriod)
{
    var Q = fusionAhrs.quaternion.element;
    var halfFeedbackError = clone(FUSION_VECTOR3_ZERO);
    if ((accelerometer.axis.x != 0.0) && (accelerometer.axis.y != 0.0) && (accelerometer.axis.z != 0.0))
    {
      var halfGravity = clone(FusionVector3);
      halfGravity.axis.x = Q.x * Q.z - Q.w * Q.y;
      halfGravity.axis.y = Q.w * Q.x + Q.y * Q.z;
      halfGravity.axis.z = Q.w * Q.w - 0.5 + Q.z * Q.z;

      halfFeedbackError = FusionVectorCrossProduct(FusionVectorFastNormalise(accelerometer), halfGravity);
      var magnetometerNorm = magnetometer.axis.x * magnetometer.axis.x
                           + magnetometer.axis.y * magnetometer.axis.y
                           + magnetometer.axis.z * magnetometer.axis.z;
      if ((magnetometerNorm > fusionAhrs.minMagneticFieldSquared) || (magnetometerNorm < fusionAhrs.maxMagneticFieldSquared))
      {
        var halfEast = clone(FusionVector3);
        halfEast.axis.x = Q.x * Q.y + Q.w * Q.z;
        halfEast.axis.y = Q.w * Q.w - 0.5 + Q.y * Q.y;
        halfEast.axis.z = Q.y * Q.z - Q.w * Q.x;

        halfFeedbackError = FusionVectorAdd(halfFeedbackError, FusionVectorCrossProduct(FusionVectorFastNormalise(FusionVectorCrossProduct(accelerometer, magnetometer)), halfEast));

        if (fusionAhrs.gain == 0)
          fusionAhrs.rampedGain = 0;

        var feedbackGain = fusionAhrs.gain;
        if (fusionAhrs.rampedGain > fusionAhrs.gain)
        {
          fusionAhrs.rampedGain -= (INITIAL_GAIN - fusionAhrs.gain) * samplePeriod / INITIALISATION_PERIOD;
          feedbackGain = fusionAhrs.rampedGain;
        }

        var halfGyroscope = FusionVectorMultiplyScalar(gyroscope, 0.5 * FUSION_DEGREES_TO_RADIANS(1));
        halfGyroscope = FusionVectorAdd(halfGyroscope, FusionVectorMultiplyScalar(halfFeedbackError, feedbackGain));
        fusionAhrs.quaternion = FusionQuaternionAdd(fusionAhrs.quaternion, FusionQuaternionMultiplyVector(fusionAhrs.quaternion, FusionVectorMultiplyScalar(halfGyroscope, samplePeriod)));
        fusionAhrs.quaternion = FusionQuaternionFastNormalise(fusionAhrs.quaternion);
        var gravity = clone(FusionVector3);
        gravity.axis.x = 2.0 * (Q.x * Q.z - Q.w * Q.y);
        gravity.axis.y = 2.0 * (Q.w * Q.x + Q.y * Q.z);
        gravity.axis.z = 2.0 * (Q.w * Q.w - 0.5 + Q.z * Q.z);
        fusionAhrs.linearAcceleration = FusionVectorSubtract(accelerometer, gravity);
      }
    }
}

function FusionAhrsCalculateEarthAcceleration(fusionAhrs)
{
  var Q = fusionAhrs.quaternion.element;
  var A = fusionAhrs.linearAcceleration.axis;
  var qwqw = Q.w * Q.w;
  var qwqx = Q.w * Q.x;
  var qwqy = Q.w * Q.y;
  var qwqz = Q.w * Q.z;
  var qxqy = Q.x * Q.y;
  var qxqz = Q.x * Q.z;
  var qyqz = Q.y * Q.z;
  var earthAcceleration = clone(FusionVector3);
  earthAcceleration.axis.x = 2.0 * ((qwqw - 0.5 + Q.x * Q.x) * A.x + (qxqy - qwqz) * A.y + (qxqz + qwqy) * A.z);
  earthAcceleration.axis.y = 2.0 * ((qxqy + qwqz) * A.x + (qwqw - 0.5 + Q.y * Q.y) * A.y + (qyqz - qwqx) * A.z);
  earthAcceleration.axis.z = 2.0 * ((qxqz - qwqy) * A.x + (qyqz + qwqx) * A.y + (qwqw - 0.5 + Q.z * Q.z) * A.z);
  return earthAcceleration;
}

function FusionAhrsIsInitialising(fusionAhrs)
{
  return fusionAhrs.rampedGain > fusionAhrs.gain;
}

function FusionAhrsReinitialise(fusionAhrs)
{
  fusionAhrs.quaternion = clone(FUSION_QUATERNION_IDENTITY);
  fusionAhrs.linearAcceleration = clone(FUSION_VECTOR3_ZERO);
  fusionAhrs.rampedGain = clone(INITIAL_GAIN);
}

function FusionAhrsZeroYaw(fusionAhrs)
{
  var Q = fusionAhrs.quaternion.element;
  fusionAhrs.quaternion = FusionQuaternionNormalise(fusionAhrs.quaternion);
  var halfInverseYaw = 0.5 * Math.atan2(Q.x * Q.y + Q.w * Q.z, Q.w * Q.w - 0.5 + Q.x * Q.x);
  var inverseYawQuaternion = clone(FusionQuaternion);
  inverseYawQuaternion.element.w = Math.cos(halfInverseYaw);
  inverseYawQuaternion.element.x = 0.0;
  inverseYawQuaternion.element.y = 0.0;
  inverseYawQuaternion.element.z = -1.0 * Math.sin(halfInverseYaw);
  fusionAhrs.quaternion = FusionQuaternionMultiply(inverseYawQuaternion, fusionAhrs.quaternion);
}
