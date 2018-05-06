import cdd
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

# Random points in 2d
np.random.seed(1)
points = np.random.randn(5, 2)
# points = np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]])

# Work with convex hull
hull = ConvexHull(points)
points_hull = hull.points[hull.vertices]
print hull.equations
"""
hull.equations is a matrix of dimension (m, d+1)

Its concatenated form is (A, -b) where the equation is Ax <= b
"""
A_hull = hull.equations[:, :-1]
b_hull = - hull.equations[:, -1]

plt.scatter(points_hull[:, 0], points_hull[:, 1])
plt.plot([0, -1.106], [3, 0])
plt.show()

# work with CDD
n = points_hull.shape[0]
data = np.hstack((np.ones((n, 1)), points_hull))
mat = cdd.Matrix(data)
mat.rep_type = cdd.RepType.GENERATOR
poly = cdd.Polyhedron(mat)
Hrep = poly.get_inequalities()
"""
Hrep is a matrix of dimension (m, d+1).

Its concatenated form is (b, -A) where the equalities is Ax <= b 
"""

A_cdd = - np.array(Hrep)[:, 1:]
b_cdd = np.array(Hrep)[:, 0]

#
print

import IPython
if IPython.get_ipython() is None:
    IPython.embed()

