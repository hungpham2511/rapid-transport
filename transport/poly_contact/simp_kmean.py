from sklearn.cluster import k_means
from .hull import ConvexHull


def simplify_kmean(hull, options=None):
    """ Simplify the convex hull using kmean algorithm.

    Parameters
    ----------
    hull: `ConvexHull`
    options: dict, optional

    Returns
    -------
    hull_out: `ConvexHull`
    """
    points = hull.vertices
    centroids, _, inertia = k_means(points, n_clusters=options['n_clusters'])
    return ConvexHull(centroids)