import imufusion
import numpy as np

samples = np.array(
    [
        [0.777, 0.277, 0.677],
        [-0.377, 0.277, 0.677],
        [0.777, -0.877, 0.677],
        [0.777, 0.277, -0.477],
        [-0.377, -0.877, 0.677],
        [-0.377, 0.277, -0.477],
        [0.777, -0.877, -0.477],
        [-0.377, -0.877, -0.477],
    ]
)

y = np.sum(samples**2, axis=1)

A = np.column_stack((samples[:, 0], samples[:, 1], samples[:, 2], np.ones_like(y)))

theta = np.linalg.pinv(A) @ y

np.set_printoptions(precision=6)

print(0.5 * theta[:3])

print(imufusion.hard_iron_solve(samples))
