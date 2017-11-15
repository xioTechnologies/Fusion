const FusionBias = '{"adcThreshold":0,"samplePeriod":0,"stationaryTimer":0,"gyroscopeBias":{"axis":{"x":0,"y":0,"z":0}}}';
const STATIONARY_PERIOD = 5.0;
const CORNER_FREQUENCY = 0.02;

function FusionBiasInitialise(fusionBias, adcThreshold, samplePeriod)
{
    fusionBias.adcThreshold = adcThreshold;
    fusionBias.samplePeriod = samplePeriod;
    fusionBias.stationaryTimer = 0.0;
    fusionBias.gyroscopeBias = clone(FUSION_VECTOR3_ZERO);
}

function FusionBiasUpdate(fusionBias, xAdc, yAdc, zAdc)
{
  if ((xAdc > fusionBias.adcThreshold) || (xAdc < (-1 * fusionBias.adcThreshold)) ||
      (yAdc > fusionBias.adcThreshold) || (yAdc < (-1 * fusionBias.adcThreshold)) ||
      (zAdc > fusionBias.adcThreshold) || (zAdc < (-1 * fusionBias.adcThreshold)))
      {
        fusionBias.stationaryTimer = 0.0;
      }
      else
      {
        if (fusionBias.stationaryTimer >= STATIONARY_PERIOD)
        {
          var gyroscope = clone(FusionVector3);
          gyroscope.axis.x = parseFloat(xAdc);
          gyroscope.axis.y = parseFloat(yAdc);
          gyroscope.axis.z = parseFloat(zAdc);
          gyroscope = FusionVectorSubtract(gyroscope, fusionBias.gyroscopeBias);
          fusionBias.gyroscopeBias = FusionVectorAdd(fusionBias.gyroscopeBias, FusionVectorMultiplyScalar(gyroscope, (2.0 * M_PI * CORNER_FREQUENCY) * fusionBias.samplePeriod));
        }
        else
        {
          fusionBias.stationaryTimer += fusionBias.samplePeriod;
        }
      }
}

function FusionBiasIsActive(fusionBias)
{
  return fusionBias.stationaryTimer >= STATIONARY_PERIOD;
}
