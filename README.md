[![Tags](https://img.shields.io/github/v/tag/xioTechnologies/Fusion.svg)](https://github.com/xioTechnologies/Fusion/tags/)
[![Build](https://img.shields.io/github/actions/workflow/status/xioTechnologies/Fusion/main.yml?branch=main)](https://github.com/xioTechnologies/Fusion/actions/workflows/main.yml)
[![Pypi](https://img.shields.io/pypi/v/imufusion.svg)](https://pypi.org/project/imufusion/)
[![Python](https://img.shields.io/pypi/pyversions/imufusion.svg)](https://pypi.org/project/imufusion/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Fusion

Fusion is a sensor fusion library for Inertial Measurement Units (IMUs), optimised for embedded systems.  Fusion is a C library but is also available as the Python package, [imufusion](https://pypi.org/project/imufusion/).  Two example Python scripts, [simple_example.py](https://github.com/xioTechnologies/Fusion/blob/main/Python/simple_example.py) and [advanced_example.py](https://github.com/xioTechnologies/Fusion/blob/main/Python/advanced_example.py) are provided with example sensor data to demonstrate use of the package.

## AHRS algorithm

The Attitude And Heading Reference System (AHRS) algorithm combines gyroscope, accelerometer, and magnetometer data into a single measurement of orientation relative to the Earth.  The algorithm also supports systems that use only a gyroscope and accelerometer, and systems that use a gyroscope and accelerometer combined with an external source of heading measurement such as GPS.

The algorithm is based on the revised AHRS algorithm presented in chapter 7 of [Madgwick's PhD thesis](https://ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.681552).  This is a different algorithm to the better-known initial AHRS algorithm presented in chapter 3, commonly referred to as the *Madgwick algorithm*.

The algorithm calculates the orientation as the integration of the gyroscope summed with a feedback term.  The feedback term is equal to the error in the current measurement of orientation as determined by the other sensors, multiplied by a gain.  The algorithm therefore functions as a complementary filter that combines high-pass filtered gyroscope measurements with low-pass filtered measurements from other sensors with a corner frequency determined by the gain.  A low gain will 'trust' the gyroscope more and so be more susceptible to drift.  A high gain will increase the influence of other sensors and the errors that result from accelerations and magnetic distortions.  A gain of zero will ignore the other sensors so that the measurement of orientation is determined by only the gyroscope.

### Initialisation

Initialisation occurs when the algorithm starts for the first time and after an acceleration rejection timeout.  During initialisation, the acceleration and magnetic rejection features are disabled and the gain is ramped down from 10 to the final value over a 3 second period.  This allows the measurement of orientation to rapidly converges from an arbitrary initial value to the value indicated by the sensors.

### Acceleration rejection

The acceleration rejection feature reduces the errors that result from the accelerations of linear and rotational motion.  Acceleration rejection works by comparing the instantaneous measurement of inclination provided by the accelerometer with the current measurement of inclination indicated by the algorithm output.  If the angular difference between these two inclinations is greater than a threshold then the accelerometer will be ignored for this algorithm update.  This is equivalent to a dynamic gain that deceases as accelerations increase.

Prolonged accelerations from linear and rotational motion may result in the accelerometer being unusable as a measurement of inclination.  This is detected by the algorithm as an acceleration rejection timeout.  An acceleration rejection timeout occurs when the number of algorithm updates that have ignored the accelerometer exceeds ten times the number that have used the accelerometer within a defined period.  If an acceleration rejection timeout occurs then the algorithm will reinitialise.

### Magnetic rejection

The magnetic rejection feature reduces the errors that result from temporary magnetic distortions.  Magnetic rejection works using the same principle as acceleration rejection operating on the magnetometer instead of the accelerometer and by comparing the measurements of heading instead of inclination.  A magnetic rejection timeout will not cause the algorithm to reinitialise.  If a magnetic rejection timeout occurs then the heading of the algorithm output will be set to the instantaneous measurement of heading provided by the magnetometer.

### Algorithm outputs

The algorithm provides three outputs: quaternion, linear acceleration, and Earth acceleration.  The quaternion describes the orientation of the sensor relative to the Earth.  This can be converted to a rotation matrix using the `FusionQuaternionToMatrix` function or to Euler angles using the `FusionQuaternionToEuler` function.  The linear acceleration is the accelerometer measurement with the 1 g of gravity removed.  The Earth acceleration is the accelerometer measurement in the Earth coordinate frame with the 1 g of gravity removed.  The algorithm supports North-West-Up (NWU), East-North-Up (ENU), and North-East-Down (NED) axes conventions.

### Algorithm settings

The AHRS algorithm settings are defined by the `FusionAhrsSettings` structure and set using the `FusionAhrsSetSettings` function.

| Setting                 | Description                                                                                                                                                                                                   |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `convention`            | Earth axes convention (NWD, ENU, or NED).                                                                                                                                                                     |
| `gain`                  | Determines the influence of the gyroscope relative to other sensors.  A value of 0.5 is appropriate for most applications.                                                                                    |
| `accelerationRejection` | Threshold (in degrees) used by the acceleration rejection feature.  A value of zero will disable this feature.  A value of 10 degrees is appropriate for most applications.                                   |
| `magneticRejection`     | Threshold (in degrees) used by the magnetic rejection feature.  A value of zero will disable the feature. A value of 20 degrees is appropriate for most applications.                                         |
| `rejectionTimeout`      | Acceleration and magnetic rejection timeout period (in samples).  A value of zero will disable the acceleration and magnetic rejection features.  A period of 5 seconds is appropriate for most applications. |

### Algorithm internal states

The AHRS algorithm internal states are defined by the `FusionAhrsInternalStates` structure and obtained using the `FusionAhrsGetInternalStates` function.

| State                        | Description                                                                                                                                                                                                                                                            |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `accelerationError`          | Angular error (in degrees) of the instantaneous measurement of inclination provided by the accelerometer.  The acceleration rejection feature will ignore the accelerometer if this value exceeds the `accelerationRejection` threshold set in the algorithm settings. |
| `accelerometerIgnored`       | `true` if the accelerometer was ignored by the previous algorithm update.                                                                                                                                                                                              |
| `accelerationRejectionTimer` | Acceleration rejection timer value normalised to between 0.0 and 1.0.  An acceleration rejection timeout will occur when this value reaches 1.0.                                                                                                                       |
| `magneticError`              | Angular error (in degrees) of the instantaneous measurement of heading provided by the magnetometer.  The magnetic rejection feature will ignore the magnetometer if this value exceeds the `magneticRejection` threshold set in the algorithm settings.               |
| `magnetometerIgnored`        | `true` if the magnetometer was ignored by the previous algorithm update.                                                                                                                                                                                               |
| `magneticRejectionTimer`     | Magnetic rejection timer value normalised to between 0.0 and 1.0.  A magnetic rejection timeout will occur when this value reaches 1.0.                                                                                                                                |

### Algorithm flags

The AHRS algorithm flags are defined by the `FusionAhrsFlags` structure and obtained using the `FusionAhrsGetFlags` function.

| Flag                           | Description                                                                                                                |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `initialising`                 | `true` if the algorithm is initialising.                                                                                   |
| `accelerationRejectionWarning` | `true` if the acceleration rejection timer has exceeded 25% of the `rejectionTimeout` value set in the algorithm settings. |
| `accelerationRejectionTimeout` | `true` if an acceleration rejection timeout has occurred and the algorithm is initialising.                                |
| `magneticRejectionWarning`     | `true` if the magnetic rejection timer has exceeded 25% of the `rejectionTimeout` value set in the algorithm settings.     |
| `magneticRejectionTimeout`     | `true` if a magnetic rejection timeout has occurred during the previous algorithm update.                                  |

## Gyroscope offset correction algorithm

The gyroscope offset correction algorithm provides run-time calibration of the gyroscope offset to compensate for variations in temperature and fine-tune existing offset calibration that may already be in place.  This algorithm should be used in conjunction with the AHRS algorithm to achieve best performance.

The algorithm calculates the gyroscope offset by detecting the stationary periods that occur naturally in most applications.  Gyroscope measurements are sampled during these periods and low-pass filtered to obtain the gyroscope offset.  The algorithm requires that gyroscope measurements do not exceed +/-3 degrees per second while stationary.  Basic gyroscope offset calibration may be necessary to ensure that the initial offset error plus measurement noise is within these bounds.

## Sensor calibration

Sensor calibration is essential for accurate measurements.  This library provides functions to apply calibration parameters to the gyroscope, accelerometer, and magnetometer.  This library does not provide a solution for calculating the calibration parameters.

### Inertial calibration

The `FusionCalibrationInertial` function applies gyroscope and accelerometer calibration parameters using the calibration model:

i<sub>c</sub> = Ms(i<sub>u</sub> - b)

- i<sub>c</sub> is the calibrated inertial measurement and `return` value
- i<sub>u</sub> is the uncalibrated inertial measurement and `uncalibrated` argument
- M is the misalignment matrix and `misalignment` argument
- s is the sensitivity diagonal matrix and `sensitivity` argument
- b is the offset vector and `offset` argument

### Magnetic calibration

The `FusionCalibrationMagnetic` function applies magnetometer calibration parameters using the calibration model:

m<sub>c</sub> = Sm<sub>u</sub> - h

- m<sub>c</sub> is the calibrated magnetometer measurement and `return` value
- m<sub>u</sub> is the uncalibrated magnetometer measurement and `uncalibrated` argument
- S is the soft iron matrix and `softIronMatrix` argument
- h is the hard iron offset vector and `hardIronOffset` argument

## Fast inverse square root

Fusion uses [Pizer's implementation](https://pizer.wordpress.com/2008/10/12/fast-inverse-square-root/) of the [fast inverse square root](https://en.wikipedia.org/wiki/Fast_inverse_square_root) algorithm for vector and quaternion normalisation.  Including the definition `FUSION_USE_NORMAL_SQRT` in [FusionMath.h](https://github.com/xioTechnologies/Fusion/blob/main/Fusion/FusionMath.h) or adding this as a preprocessor definition will use normal square root operations for all normalisation calculations.  This will slow down execution speed for a small increase in accuracy.  The increase in accuracy will typically be too small to observe on any practical system.
