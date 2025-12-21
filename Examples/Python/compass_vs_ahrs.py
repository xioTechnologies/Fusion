import sys

import imufusion
import matplotlib.pyplot as plt
import numpy as np

# Import sensor data
data = np.genfromtxt("sensor_data.csv", delimiter=",", skip_header=1)

sample_rate = 100  # 100 Hz

seconds = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]
magnetometer = data[:, 7:10]

# Compass heading
compass_heading = [imufusion.compass(a, m) for a, m in zip(accelerometer, magnetometer)]

# AHRS heading
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.AhrsSettings(
    magnetic_rejection=10,  # reject magnetic disturbances >10 degrees
    recovery_trigger_period=20 * sample_rate,  # reject magnetic disturbances for up to 20 seconds
)

ahrs_heading = np.empty_like(seconds)

for index, _ in enumerate(seconds):
    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], 1 / sample_rate)

    ahrs_heading[index] = imufusion.quaternion_to_euler(ahrs.quaternion)[2]

# Plot
plt.title("Heading")
plt.plot(seconds, compass_heading, label="Compass")
plt.plot(seconds, ahrs_heading, label="AHRS")
plt.xlabel("Seconds")
plt.ylabel("Degrees")
plt.legend()
plt.grid()

plt.show(block="dont_block" not in sys.argv)  # don't block when script run by CI
