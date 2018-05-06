from .plannar_geometry import line_intersect
import numpy as np


def halfplanes_intersection(a, b, c, xlow=0.0, xhigh=100, ulow=-1000.0, uhigh=1000):
    """ Halfplanes intersection algorithm.

    Notes
    -----
    Initial polygon is constructed from the given fixed bound.

    Parameters
    ----------
    a
    b
    c
    xlow
    xhigh
    ulow
    uhigh

    Returns
    -------

    """
    N_halfplane = len(a)
    a_bnd = np.array([1, -1, 0, 0], dtype=float)
    b_bnd = np.array([0, 0, 1, -1], dtype=float)
    c_bnd = np.array([-uhigh, ulow, -xhigh, xlow], dtype=float)
    edges = [N_halfplane, N_halfplane+3, N_halfplane+1, N_halfplane+2]
    for i in range(N_halfplane):
        N_edge = len(edges)

        # Polgon vertices
        points = []
        for k in range(N_edge):
            e1 = edges[k]
            e2 = edges[(k + 1) % N_edge]
            if e1 <= N_halfplane - 1:
                l1 = [a[e1], b[e1], c[e1]]
            else:
                l1 = [a_bnd[e1 - N_halfplane], b_bnd[e1 - N_halfplane], c_bnd[e1 - N_halfplane]]
            if e2 <= N_halfplane - 1:
                l2 = [a[e2], b[e2], c[e2]]
            else:
                l2 = [a_bnd[e2 - N_halfplane], b_bnd[e2 - N_halfplane], c_bnd[e2 - N_halfplane]]
            points.append(line_intersect(l1, l2))

        # Process points which satisfies constraints
        N_points_sat = 0
        found_unsat = False  # True when find the first point that does not satisfy line i
        sat_start_index = -2  # Index of the first point that satisfies line i
        for k in range(N_edge):
            u, x = points[k]
            if a[i] * u + b[i] * x + c[i] > 1e-9:
                found_unsat = True
            else:
                N_points_sat += 1
                if found_unsat and sat_start_index == -2:
                    sat_start_index = k

        if sat_start_index == -2:  # Case: True, True, False, False, ...., False
            sat_start_index = 0

        # Post-processing
        if N_points_sat == N_edge:
            continue
        elif N_points_sat == 0:
            return []
        edges_new = []
        for k in range(N_points_sat + 1):
            edges_new.append(edges[(sat_start_index + k) % N_edge])
        edges_new.append(i)
        edges = edges_new
    edges = filter(lambda s: s < N_halfplane, edges)
    return edges

