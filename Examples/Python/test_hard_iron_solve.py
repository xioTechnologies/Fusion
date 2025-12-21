import imufusion
import numpy as np

samples = np.random.normal(size=(100, 3))

samples /= np.linalg.norm(samples, axis=1)[:, None]  # normalise to unit sphere

samples *= 1.0 + 0.01 * np.random.normal(size=samples.shape)  # add noise

samples += np.array([0.1, 0.2, 0.3])  # add hard-iron offset

y = np.sum(samples**2, axis=1)

A = np.column_stack((samples[:, 0], samples[:, 1], samples[:, 2], np.ones_like(y)))

theta = np.linalg.pinv(A) @ y

numpy_h = 0.5 * theta[:3]

fusion_h = imufusion.HardIron.solve(samples)

assert np.allclose(numpy_h, fusion_h, atol=1e-6), f"{numpy_h} != {fusion_h}"
