import imufusion
import numpy as np

for _ in range(100):
    # Inertial sensor model
    uncalibrated = np.random.rand(3)
    misalignment = np.random.rand(3, 3)
    sensitivity = np.random.rand(3)
    offset = np.random.rand(3)

    numpy_inertial = misalignment @ ((uncalibrated - offset) * sensitivity)

    fusion_inertial = imufusion.model_inertial(uncalibrated, misalignment, sensitivity, offset)

    assert np.allclose(numpy_inertial, fusion_inertial, atol=1e-6), f"{numpy_inertial} != {fusion_inertial}"

    # Magnetic sensor model
    uncalibrated = np.random.rand(3)
    soft_iron = np.random.rand(3, 3)
    hard_iron = np.random.rand(3)

    numpy_magnetic = soft_iron @ (uncalibrated - hard_iron)

    fusion_magnetic = imufusion.model_magnetic(uncalibrated, soft_iron, hard_iron)

    assert np.allclose(numpy_magnetic, fusion_magnetic, atol=1e-6), f"{numpy_magnetic} != {fusion_magnetic}"
