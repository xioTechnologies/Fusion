import os
import sys

import imufusion
import matplotlib.pyplot as plt
import numpy as np

# Read sensor data CSV
examples_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(examples_directory, "Sensor Data.csv")

sensor_data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

seconds = sensor_data[:, 0]
gyroscope = sensor_data[:, 1:4]
accelerometer = sensor_data[:, 4:7]
magnetometer = sensor_data[:, 7:10]

# Compass heading
compass_heading = [imufusion.compass(a, m) for a, m in zip(accelerometer, magnetometer)]

# AHRS heading
sample_rate = 100  # Hz

ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.AhrsSettings(
    sample_rate,
    magnetic_rejection=10,  # reject magnetic errors >10 degrees
    recovery_trigger_period=int(20 * sample_rate),  # reject magnetic disturbances for up to 20 seconds
)

ahrs_heading = np.empty_like(seconds)

for index, _ in enumerate(seconds):
    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index])

    ahrs_heading[index] = imufusion.quaternion_to_euler(ahrs.quaternion)[2]

# Plot
plt.plot(seconds, compass_heading, label="Compass")
plt.plot(seconds, ahrs_heading, label="AHRS")
plt.title("Heading")
plt.xlabel("Seconds")
plt.ylabel("Degrees")
plt.legend()
plt.grid()

plt.show(block="dont_block" not in sys.argv)  # don't block when script run by CI
