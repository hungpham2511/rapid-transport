import toppra_app
import numpy as np


def test_basic_plannar():
    pts = [[0, 0], [1, 0], [1, 1]]
    hull = toppra_app.poly_contact.ConvexHull(pts)

    sample = toppra_app.poly_contact.uniform_face_sampling(hull)
    A, b = hull.get_halfspaces()
    np.testing.assert_array_less(A.dot(sample), b + 1e-9)  # Satisfy constraint
    assert np.any(np.abs(A.dot(sample) - b) < 1e-9)  # On face
