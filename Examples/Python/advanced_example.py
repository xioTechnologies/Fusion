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

sample_rate = 100  # Hz

# Configure AHRS algorithm
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.AhrsSettings(
    sample_rate,
    convention=imufusion.CONVENTION_NWU,
    gain=0.5,
    gyroscope_range=2000,
    acceleration_rejection=10,
    magnetic_rejection=10,
    recovery_trigger_period=int(5 * sample_rate),  # 5 seconds
)

# Configure bias algorithm
bias = imufusion.Bias()

bias.settings = imufusion.BiasSettings(sample_rate)

# Process each CSV line as if reading sensor data in real-time
delta_time = np.diff(seconds, prepend=seconds[0])

euler = np.empty((len(seconds), 3))
internal_states = np.empty((len(seconds), 6))
flags = np.empty((len(seconds), 4))

for index, _ in enumerate(seconds):
    # Calibration parameters
    gyroscope_misalignment = np.identity(3)
    gyroscope_sensitivity = np.ones(3)
    gyroscope_offset = np.zeros(3)

    accelerometer_misalignment = np.identity(3)
    accelerometer_sensitivity = np.ones(3)
    accelerometer_offset = np.zeros(3)

    soft_iron_matrix = np.identity(3)
    hard_iron_offset = np.zeros(3)

    # Apply calibration
    gyroscope[index] = imufusion.model_inertial(
        gyroscope[index],
        gyroscope_misalignment,
        gyroscope_sensitivity,
        gyroscope_offset,
    )

    accelerometer[index] = imufusion.model_inertial(
        accelerometer[index],
        accelerometer_misalignment,
        accelerometer_sensitivity,
        accelerometer_offset,
    )

    magnetometer[index] = imufusion.model_magnetic(
        magnetometer[index],
        soft_iron_matrix,
        hard_iron_offset,
    )

    # Update bias algorithm
    gyroscope[index] = bias.update(gyroscope[index])

    # Update AHRS algorithm
    ahrs.sample_period = delta_time[index]

    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index])

    # ...
    euler[index] = imufusion.quaternion_to_euler(ahrs.quaternion)

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = np.array(
        [
            ahrs_internal_states.acceleration_error,
            ahrs_internal_states.accelerometer_ignored,
            ahrs_internal_states.acceleration_recovery_trigger,
            ahrs_internal_states.magnetic_error,
            ahrs_internal_states.magnetometer_ignored,
            ahrs_internal_states.magnetic_recovery_trigger,
        ]
    )

    ahrs_flags = ahrs.flags
    flags[index] = np.array(
        [
            ahrs_flags.initialising,
            ahrs_flags.angular_rate_recovery,
            ahrs_flags.acceleration_recovery,
            ahrs_flags.magnetic_recovery,
        ]
    )


def plot_bool(axis, x, y, label):
    axis.plot(x, y, "tab:cyan", label=label)
    plt.sca(axis)
    plt.yticks([0, 1], ["False", "True"])
    axis.grid()
    axis.legend()


# Plot Euler angles
figure, axes = plt.subplots(nrows=11, sharex=True, gridspec_kw={"height_ratios": [6, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]})

figure.suptitle("Euler angles, internal states, and flags")

axes[0].plot(seconds, euler[:, 0], "tab:red", label="Roll")
axes[0].plot(seconds, euler[:, 1], "tab:green", label="Pitch")
axes[0].plot(seconds, euler[:, 2], "tab:blue", label="Yaw")
axes[0].set_ylabel("Degrees")
axes[0].grid()
axes[0].legend()

# Plot initialising flag
plot_bool(axes[1], seconds, flags[:, 0], "Initialising")

# Plot angular rate recovery flag
plot_bool(axes[2], seconds, flags[:, 1], "Angular rate recovery")

# Plot acceleration rejection internal states and flag
axes[3].plot(seconds, internal_states[:, 0], "tab:olive", label="Acceleration error")
axes[3].set_ylabel("Degrees")
axes[3].grid()
axes[3].legend()

plot_bool(axes[4], seconds, internal_states[:, 1], "Accelerometer ignored")

axes[5].plot(seconds, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
axes[5].grid()
axes[5].legend()

plot_bool(axes[6], seconds, flags[:, 2], "Acceleration recovery")

# Plot magnetic rejection internal states and flag
axes[7].plot(seconds, internal_states[:, 3], "tab:olive", label="Magnetic error")
axes[7].set_ylabel("Degrees")
axes[7].grid()
axes[7].legend()

plot_bool(axes[8], seconds, internal_states[:, 4], "Magnetometer ignored")

axes[9].plot(seconds, internal_states[:, 5], "tab:orange", label="Magnetic recovery trigger")
axes[9].grid()
axes[9].legend()

plot_bool(axes[10], seconds, flags[:, 3], "Magnetic recovery")

plt.show(block="dont_block" not in sys.argv)  # don't block when script run by CI
