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

# Configure AHRS algorithm
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.AhrsSettings(sample_rate=100)  # Hz

# Process each CSV line as if reading sensor data in real-time
euler = np.empty((len(seconds), 3))

for index, _ in enumerate(seconds):
    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index])

    euler[index] = imufusion.quaternion_to_euler(ahrs.quaternion)

# Plot sensor data and Euler angles
_, axes = plt.subplots(nrows=3, sharex=True)


def plot_xyz(axis, x, y, title, units, legend=("X", "Y", "Z")):
    axis.plot(x, y[:, 0], "tab:red", label=legend[0])
    axis.plot(x, y[:, 1], "tab:green", label=legend[1])
    axis.plot(x, y[:, 2], "tab:blue", label=legend[2])
    axis.set_title(title)
    axis.set_ylabel(units)
    axis.grid()
    axis.legend()


plot_xyz(axes[0], seconds, gyroscope, "Gyroscope", "Degrees/s")

plot_xyz(axes[1], seconds, accelerometer, "Accelerometer", "Degrees/s")

plot_xyz(axes[2], seconds, euler, "Euler angles", "Degrees", ("Roll", "Pitch", "Yaw"))

plt.show(block="dont_block" not in sys.argv)  # don't block when script run by CI
