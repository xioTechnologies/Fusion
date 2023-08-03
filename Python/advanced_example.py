import imufusion
import matplotlib.pyplot as pyplot
import numpy
import sys

# Import sensor data
data = numpy.genfromtxt("sensor_data.csv", delimiter=",", skip_header=1)

sample_rate = 100  # 100 Hz

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]
magnetometer = data[:, 7:10]

# Instantiate algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,  # convention
                                   0.5,  # gain
                                   10,  # acceleration rejection
                                   20,  # magnetic rejection
                                   5 * sample_rate)  # recovery trigger period = 5 seconds

# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 6))
flags = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_recovery_trigger,
                                          ahrs_internal_states.magnetic_error,
                                          ahrs_internal_states.magnetometer_ignored,
                                          ahrs_internal_states.magnetic_recovery_trigger])

    ahrs_flags = ahrs.flags
    flags[index] = numpy.array([ahrs_flags.initialising,
                                ahrs_flags.acceleration_recovery,
                                ahrs_flags.magnetic_recovery])


def plot_bool(axis, x, y, label):
    axis.plot(x, y, "tab:cyan", label=label)
    pyplot.sca(axis)
    pyplot.yticks([0, 1], ["False", "True"])
    axis.grid()
    axis.legend()


# Plot Euler angles
figure, axes = pyplot.subplots(nrows=10, sharex=True, gridspec_kw={"height_ratios": [6, 1, 2, 1, 1, 1, 2, 1, 1, 1]})

figure.suptitle("Euler angles, internal states, and flags")

axes[0].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[0].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[0].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes[0].set_ylabel("Degrees")
axes[0].grid()
axes[0].legend()

# Plot initialising flag
plot_bool(axes[1], timestamp, flags[:, 0], "Initialising")

# Plot acceleration rejection internal states and flags
axes[2].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
axes[2].set_ylabel("Degrees")
axes[2].grid()
axes[2].legend()

plot_bool(axes[3], timestamp, internal_states[:, 1], "Accelerometer ignored")

axes[4].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
axes[4].grid()
axes[4].legend()

plot_bool(axes[5], timestamp, flags[:, 1], "Acceleration recovery")

# Plot magnetic rejection internal states and flags
axes[6].plot(timestamp, internal_states[:, 3], "tab:olive", label="Magnetic error")
axes[6].set_ylabel("Degrees")
axes[6].grid()
axes[6].legend()

plot_bool(axes[7], timestamp, internal_states[:, 4], "Magnetometer ignored")

axes[8].plot(timestamp, internal_states[:, 5], "tab:orange", label="Magnetic recovery trigger")
axes[8].grid()
axes[8].legend()

plot_bool(axes[9], timestamp, flags[:, 2], "Magnetic recovery")

if len(sys.argv) == 1:  # don't show plots when script run by CI
    pyplot.show()
