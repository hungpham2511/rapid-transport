from ..utils import compute_area_ndim
import numpy as np


def uniform_face_sampling(hull):
    """ Return a point sampled randomly from hull's faces.

    Parameters
    ----------
    hull: toppra_app.ConvexHull

    Returns
    -------
    points: (n,)array
        Sampled point.
    """
    dof = hull.get_dim()
    faces = hull.get_faces()
    N_faces = faces.shape[0]
    faces_area = hull.compute_face_areas()
    faces_prob = faces_area / np.sum(np.abs(faces_area))
    face_sel = faces[np.random.choice(N_faces, p=faces_prob)]
    simplex_prob = np.random.dirichlet(np.ones(dof))

    point =  face_sel.T.dot(simplex_prob)
    return point


def uniform_interior_sampling(hull):
     """ Return a point sampled randomly from the hull's interior.

    Parameters
    ----------
    hull: toppra_app.ConvexHull

    Returns
    -------
    points: (n,)array
        Sampled point.
    """
     tri_area = hull.get_simplex_areas()
     tri_prob = tri_area / np.sum(tri_area)
     simplex_sel = hull.get_simplices()[np.random.choice(tri_area.shape[0], p=tri_prob)]
     # Now, sample from the simplex. This is done by first sampling from the Dirichlet
     # distribution with alpha=1, so that one obtains a uniform distribution from the
     # probability simplex. Then, map it to a point via the convex combination formula.
     # The reason why this works is, according to the answer by Hendrik and Douglas Zare
     # in this SOF answer:
     # https://stackoverflow.com/questions/24800820/random-vectors-uniformely-distributed-into-convex-n-polytope?noredirect=1#comment38496115_24800820
     # the invertible linear transformation preserves uniform probability distribution.
     alphas = np.random.dirichlet(np.ones(hull.get_dim() + 1))
     sample = simplex_sel.T.dot(alphas)
     return sample

