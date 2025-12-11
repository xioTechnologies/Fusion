import imufusion
import numpy as np
import scipy

quaternions = scipy.spatial.transform.Rotation.random(10000).as_quat()

for quaternion in quaternions:
    # Quaternion to matrix
    scipy_matrix = scipy.spatial.transform.Rotation.from_quat(quaternion).as_matrix()

    fusion_matrix = imufusion.Quaternion(quaternion[[3, 0, 1, 2]]).to_matrix()

    assert np.allclose(scipy_matrix, fusion_matrix, atol=1e-6), f"{scipy_matrix} != {fusion_matrix}"

    # Quaternion to Euler
    scipy_euler = scipy.spatial.transform.Rotation.from_quat(quaternion).as_euler("xyz", degrees=True)

    fusion_euler = imufusion.Quaternion(quaternion[[3, 0, 1, 2]]).to_euler()

    if np.abs(scipy_euler[1]) < 89:  # avoid gimbal-lock region where Euler angles are undefined
        assert np.allclose(scipy_euler, fusion_euler, atol=1e-3), f"{scipy_euler} != {fusion_euler}"
