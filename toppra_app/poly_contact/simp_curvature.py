from ..utils import generate_random_unit_vectors
from .hull import ConvexHull
import numpy as np


def simplify_curvature(hull, options):
    """ Pixel Purity Index algorithm.

    An algorithm called  seems to resemble this technique.

    Parameters
    ----------
    hull: ConvexHull
        The Convex Hull to simplify.
    options: dict
        Options include number of random vectors, number of extremes to get.

    Returns
    -------
    ConvexHull
    """
    dof = hull.get_vertices().shape[1]
    unit_vectors = generate_random_unit_vectors(dof, options['n_unit_vectors'])
    PP_indices = np.zeros(hull.get_vertices().shape[0])
    for vector in unit_vectors:
        dot_prod = hull.get_vertices().dot(vector)
        PP_indices[np.argmax(dot_prod)] += 1
    PP_indices = PP_indices / options['n_unit_vectors']
    ind = np.argsort(PP_indices)[-options['n_extremes']:]
    return ConvexHull(hull.get_vertices()[ind])
