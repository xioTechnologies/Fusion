/**
 * @file FusionCompass.h
 * @author Seb Madgwick
 * @brief Tilt-compensated compass to calculate an heading relative to magnetic
 * north using accelerometer and magnetometer measurements.
 */

#ifndef FUSION_COMPASS_H
#define FUSION_COMPASS_H

//------------------------------------------------------------------------------
// Includes

#include "FusionMath.h"

//------------------------------------------------------------------------------
// Function declarations

float FusionCompassCalculateHeading(const FusionVector accelerometer, const FusionVector magnetometer);

#endif

//------------------------------------------------------------------------------
// End of file
