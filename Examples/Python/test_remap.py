import imufusion
import numpy as np
import scipy

rotations = scipy.spatial.transform.Rotation.create_group("O")

rotations = sorted(rotations, key=lambda r: tuple(r.as_matrix().astype(int).flatten()))

rotations.reverse()

for index, rotation in enumerate(rotations):
    sensor = np.random.rand(3)

    scipy_sensor = rotation.as_matrix() @ sensor

    fusion_sensor = imufusion.remap(sensor, index)

    assert np.allclose(scipy_sensor, fusion_sensor, atol=1e-6), f"{scipy_sensor} != {fusion_sensor}"
