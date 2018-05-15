from ..utils import generate_random_unit_vectors, compute_area_ndim
from .hull import ConvexHull
from scipy.spatial import ConvexHull as sciConvexHull
from random import shuffle
import numpy as np
import scipy.linalg
import logging
logger = logging.getLogger(__name__)


def vertex_component_analysis(points):
    """ Vertex component analysis algorithm.

    Parameters
    ----------
    points: (N,d)array
        Data points.

    Returns
    -------
    simplex_indices: list of int
        Indices of the VCA simplex's corners.
    """
    dof = len(points[0])
    N_simplex = dof + 1  # Number of vertex in the base simplex

    valid_simplex = False
    while not valid_simplex:
        unit_vectors = generate_random_unit_vectors(dof, N_simplex)
        vertices = []  # List of arrays
        vertices_indices = []   # Indices of vertices
        for i in range(N_simplex):
            unit_vector = unit_vectors[i]

            # If there are more than two vertices: project along the normal direction
            if i >= 2:
                A = (np.array(vertices[1:]) - np.array(vertices[0])).T
                # unit_vector_projected = A.dot(np.linalg.inv(A.T.dot(A))).dot(A.T).dot(unit_vector)
                unit_vector_projected = A.dot(scipy.linalg.pinv(A)).dot(unit_vector)
                unit_vector = unit_vector - unit_vector_projected
            vertices_indices.append(np.argmax(points.dot(unit_vector)))
            vertices.append(points[vertices_indices[-1]])
        try:
            sciConvexHull(vertices)
            valid_simplex = True
        except Exception as e:
            print(e)
    return vertices_indices


def poly_expansion(points, seed_indices, N_pt_max, max_area=True):
    """ Expand along faces to build up a new polyhedron.

    Parameters
    ----------
    points: (N,d)array
        Data points. These need not be vertices of a convex hull.
    seed_indices: list of int
        Indices of the seed vertices.
    N_pt_max: int
    max_area: bool, optional

    Returns
    -------
    vertices_indices: (M,)int array
        Indices of the vertices.
        Can be None if the operation fails.
    """
    vertices = points[seed_indices].tolist()
    vertices_indices = list(seed_indices)
    centroid = np.sum(vertices, axis=0) / len(seed_indices)

    expanded_faces = []  # Indices of faces that has been expanded

    # Iterative expansion
    N_vertices = len(vertices_indices)
    while N_vertices < N_pt_max:
        logger.debug("Expanding, number of vertices {:d}".format(len(vertices_indices)))
        try:
            hull_ = sciConvexHull(vertices)
            N_vertices = len(hull_.vertices)
        except:
            return None

        if max_area:
            simplex_max_area = hull_.simplices[0]
            max_area = -1
            for simplex in hull_.simplices:
                area = compute_area_ndim(hull_.points[simplex])
                simplex_indices = [vertices_indices[j] for j in simplex]
                if area > max_area and set(simplex_indices) not in expanded_faces:
                    simplex_max_area = simplex
                    max_area = area
            assert max_area != -1,  "Unable to find a face for expansion!"
            simplex_indices = [vertices_indices[j] for j in simplex_max_area]
            vertices_on_face = hull_.points[simplex_max_area]  # Coordinates

        else:  # Random selection
            random_list = range(len(hull_.simplices))
            shuffle(random_list)
            for j in random_list:
                simplex = hull_.simplices[j]
                simplex_indices = [vertices_indices[j] for j in simplex]
                if set(simplex_indices) not in expanded_faces:
                    vertices_on_face = points[simplex_indices]  # Coordinates
                    break
        expanded_faces.append(set(simplex_indices))

        face_centroid = np.sum(vertices_on_face, axis=0)
        proj_vector = face_centroid - centroid
        A = (vertices_on_face[1:] - vertices_on_face[0]).T
        proj_vector_ortho = proj_vector - A.dot(scipy.linalg.pinv(A)).dot(proj_vector)
        opt_vertex_index = np.argmax(points.dot(proj_vector_ortho))
        if opt_vertex_index in vertices_indices:
            # print "face {:}, already found vertex {:d}".format(simplex_indices, opt_vertex_index)
            continue
        else:
            vertices_indices.append(opt_vertex_index)
            vertices.append(points[opt_vertex_index])
    return vertices_indices


def simplify_vertex_analysis(hull, options):
    """ A variant of the Vertex Component Analysis algorithm.

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

    dof = hull.get_dim()  # space dimension
    N_simplex = dof + 1  # Number of vertex in the base simplex

    unit_vectors = generate_random_unit_vectors(dof, N_simplex)

    vertices = []  # List of arrays
    vertices_indices = []   # Indices of vertices in hull.vertices matrix
    for i in range(N_simplex):
        unit_vector = unit_vectors[i]

        # If there are more than two vertices: project along the normal direction
        if i >= 2:
            A = (np.array(vertices[1:]) - np.array(vertices[0])).T
            # unit_vector_projected = A.dot(np.linalg.inv(A.T.dot(A))).dot(A.T).dot(unit_vector)
            unit_vector_projected = A.dot(scipy.linalg.pinv(A)).dot(unit_vector)
            unit_vector = unit_vector - unit_vector_projected
        vertices_indices.append(np.argmax(hull.vertices.dot(unit_vector)))
        opt_vertex = hull.vertices[vertices_indices[-1]]
        vertices.append(opt_vertex)

    centroid = np.sum(vertices, axis=0) / (N_simplex)

    # Faces of the simplex
    faces = []  # All discovered faces
    for i in vertices_indices:
        simplex_face_indices = filter(lambda k: k != i, vertices_indices)
        faces.append(set(simplex_face_indices))

    expanded_faces = []  # Indices of faces that has been expanded

    # Iterative expansion
    for i in range(options['max_iter']):
        try:
            hull_ = sciConvexHull(vertices)
        except:
            return None

        if options['max_area']:
            simplex_max_area = hull_.simplices[0]
            max_area = -1
            for simplex in hull_.simplices:
                area = compute_area_ndim(hull_.points[simplex])
                simplex_indices = [vertices_indices[j] for j in simplex]
                if area > max_area and set(simplex_indices) not in expanded_faces:
                    simplex_max_area = simplex
                    max_area = area
            assert max_area != -1,  "Unable to find a face for expansion!"

            simplex_indices = [vertices_indices[j] for j in simplex_max_area]
            vertices_on_face = hull_.points[simplex_max_area]  # Coordinates

        else:  # Random selection
            random_list = range(len(hull_.simplices))
            shuffle(random_list)
            for j in random_list:
                simplex = hull_.simplices[j]
                simplex_indices = [vertices_indices[j] for j in simplex]
                if set(simplex_indices) not in expanded_faces:
                    vertices_on_face = hull.vertices[simplex_indices]  # Coordinates
                    break
        expanded_faces.append(set(simplex_indices))

        face_centroid = np.sum(vertices_on_face, axis=0)
        proj_vector = face_centroid - centroid
        A = (vertices_on_face[1:] - vertices_on_face[0]).T
        proj_vector_ortho = proj_vector - A.dot(scipy.linalg.pinv(A)).dot(proj_vector)
        opt_vertex_index = np.argmax(hull.vertices.dot(proj_vector_ortho))
        if opt_vertex_index in vertices_indices:
            # print "face {:}, already found vertex {:d}".format(simplex_indices, opt_vertex_index)
            continue
        else:
            vertices_indices.append(opt_vertex_index)
            vertices.append(hull.vertices[opt_vertex_index])

    return ConvexHull(vertices)





