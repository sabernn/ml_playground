

import numpy as np
from scipy.linalg import eig, sqrtm

def fit_ellipsoid(points):
    # Convert points to homogeneous coordinates
    points = np.column_stack((points, np.ones(len(points))))

    # Solve the linear equation for the ellipsoid parameters
    D = np.array([
        points[:,0] * points[:,0],
        points[:,1] * points[:,1],
        points[:,2] * points[:,2],
        2 * points[:,0] * points[:,1],
        2 * points[:,0] * points[:,2],
        2 * points[:,1] * points[:,2],
        2 * points[:,0],
        2 * points[:,1],
        2 * points[:,2]
    ]).T

    S = np.dot(D.T, D)
    C = np.zeros([9, 9])
    C[0, 4] = C[4, 0] = 2
    C[1, 1] = -1
    C[2, 2] = -1
    C[3, 3] = -1
    C[4, 4] = -1
    C[5, 5] = -1

    eigenvalues, eigenvectors = eig(np.dot(np.linalg.inv(S), C))
    radii = np.sqrt(1 / np.abs(eigenvalues))
    transformation = sqrtm(np.diag(radii)) @ eigenvectors

    return transformation[:3, :]

# Example usage
points = np.array([
    [1, 2, 38],
    [4, 50, 6],
    [-7, 8, 9],
    [10, -11, 12]
])

transformation = fit_ellipsoid(points)
print(transformation)
