import sys

import imufusion
import matplotlib.pyplot as plt
import numpy as np

# Import sensor data
data = np.genfromtxt("sensor_data.csv", delimiter=",", skip_header=1)

sample_rate = 100  # 100 Hz

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]
magnetometer = data[:, 7:10]

# Instantiate algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.AhrsSettings(
    imufusion.CONVENTION_NWU,  # convention
    0.5,  # gain
    2000,  # gyroscope range
    10,  # acceleration rejection
    10,  # magnetic rejection
    5 * sample_rate,  # recovery trigger period = 5 seconds
)

# Process sensor data
delta_time = np.diff(timestamp, prepend=timestamp[0])

euler = np.empty((len(timestamp), 3))
internal_states = np.empty((len(timestamp), 6))
flags = np.empty((len(timestamp), 4))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])

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

axes[0].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[0].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[0].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes[0].set_ylabel("Degrees")
axes[0].grid()
axes[0].legend()

# Plot initialising flag
plot_bool(axes[1], timestamp, flags[:, 0], "Initialising")

# Plot angular rate recovery flag
plot_bool(axes[2], timestamp, flags[:, 1], "Angular rate recovery")

# Plot acceleration rejection internal states and flag
axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
axes[3].set_ylabel("Degrees")
axes[3].grid()
axes[3].legend()

plot_bool(axes[4], timestamp, internal_states[:, 1], "Accelerometer ignored")

axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
axes[5].grid()
axes[5].legend()

plot_bool(axes[6], timestamp, flags[:, 2], "Acceleration recovery")

# Plot magnetic rejection internal states and flag
axes[7].plot(timestamp, internal_states[:, 3], "tab:olive", label="Magnetic error")
axes[7].set_ylabel("Degrees")
axes[7].grid()
axes[7].legend()

plot_bool(axes[8], timestamp, internal_states[:, 4], "Magnetometer ignored")

axes[9].plot(timestamp, internal_states[:, 5], "tab:orange", label="Magnetic recovery trigger")
axes[9].grid()
axes[9].legend()

plot_bool(axes[10], timestamp, flags[:, 3], "Magnetic recovery")

plt.show(block="dont_block" not in sys.argv)  # don't block when script run by CI
