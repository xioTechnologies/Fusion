function FusionCompassCalculateHeading(accelerometer, magnetometer)
{
  var magneticWest = FusionVectorFastNormalise(FusionVectorCrossProduct(accelerometer, magnetometer));
  var magneticNorth = FusionVectorFastNormalise(FusionVectorCrossProduct(magneticWest, accelerometer));
  return FUSION_RADIANS_TO_DEGREES(Math.atan2(magneticWest.axis.x, magneticNorth.axis.x));
}
