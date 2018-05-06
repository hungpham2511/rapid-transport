import scipy.spatial
import os, logging
import numpy as np
import cvxpy as cvx
from ..utils import sample_uniform, compute_volume_simplex, line_prepender, compute_area_ndim

logger = logging.getLogger(__name__)


class ConvexHull(object):
    """ ConvexHull of wrench data.

    Contain some helper methods extending the functionality of scipy.spatial.ConvexHull.
    In any case, this class is a wrapper about scipy's implementation.

    Parameters
    ----------
    points: (n,6)array
        Coordinate of wrench data.
    faces: (array, array) or None
        If provided, convex hull will not be generated, and algorithm will be used instead.
    """
    def __init__(self, points, faces=None):
        self.dim = len(points[0])
        self._id = str(np.abs(int(13213214378 * np.sum(points))) % 58329)
        if faces is None:
            self.hull = scipy.spatial.ConvexHull(points)
            self.A = self.hull.equations[:, :-1]
            self.b = - self.hull.equations[:, -1]
            self.vertices = self.hull.points[self.hull.vertices]
            self.faces = self.hull.points[self.hull.simplices]
            self.face_areas = None
            try:
                file_ = np.load(os.path.expanduser("~/.temp/" + self._id + ".npz"))
                logger.info("Found saved triangulation data. Loading from ~/.temp/{:}.npz".format(self._id))
                self._simplices = file_['simplices']
                self._simplex_areas = file_['simplex_areas']
            except IOError:
                self._simplices = None
                self._simplex_areas = None
        else:
            self.vertices = np.array(points)
            self.A = faces[0]
            self.b = faces[1]

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        """ Return all faces, which are lists of points.

        Returns
        -------
        out: (N,dof,dof)array
            N faces.
        """
        return self.faces

    def compute_face_areas(self):
        if self.face_areas is None:
            self.face_areas = []
            for face in self.faces:
                self.face_areas.append(compute_area_ndim(face))
            self.face_areas = np.array(self.face_areas)
        return self.face_areas

    def get_simplices(self):
        if self._simplices is None:
            logger.debug("Triangulation not found! Start triangulating.")
            self.delaunay = scipy.spatial.Delaunay(self.vertices)
            self._simplices = self.delaunay.points[self.delaunay.simplices]
            self._simplex_areas = []
            logger.debug("Computing the volumes of the simplices.")
            N_simples = self._simplices.shape[0]
            for i, simplex in enumerate(self._simplices):
                self._simplex_areas.append(compute_volume_simplex(simplex))
            self._simplex_areas = np.array(self._simplex_areas)
            np.savez(os.path.expanduser("~/.temp/" + self._id + ".npz"),
                     simplices=self._simplices, simplex_areas=self._simplex_areas)
            logger.info("Save triangulation data to ~/.temp/{:}.npz".format(self._id))
        return self._simplices

    def get_simplex_areas(self):
        if self._simplex_areas is None:
            self.get_simplices()
        return self._simplex_areas

    def get_halfspaces(self):
        """ Coefficient for the half-spaces constraints Ax <= b.

        Returns
        -------
        A: array
        b: array
        """
        return self.A, self.b

    def get_dim(self):
        return self.dim

    def compute_volume(self, method='qhull'):
        if method == 'triangulation':
            return self._compute_volume_delaunay()
        elif method == 'monte':
            return self._compute_volume_monte()
        elif method == 'qhull':  # The only working solution
            return self._compute_volume_qhull()
        else:
            raise NotImplementedError

    def report(self):
        report_string = "Convex Hull report\n -----------------\n"
        report_string += "    Volume: {:f}\n".format(self.compute_volume())
        report_string += "    No. Vertices: {:d}\n".format(self.vertices.shape[0])
        report_string += "    No. Faces: {:d}\n".format(self.A.shape[0])
        return report_string

    def validate(self, N_vectors=1000, eps=1e-10):
        """ Validate that the two fields `vertices` and `halfspaces` are consistent.

        The following strategy is employed:

        - Generate N random vectors
        - For each vector, do two optimizations:
        - find the point in `vertices` that optimizes the dot product with the vector,
        - optimize v^T x s.t. Ax <= b
        - Validation success if the two objectives are similar.

        Returns
        -------
        validated: bool
        msg: string
            Output Message.
        """
        if self.A.shape[1] != self.dim:
            return False, "Wrong dimension!"
        if self.A.shape[0] != self.b.shape[0]:
            return False, "Halfspace equation inconsistent!"
        for i in range(N_vectors):
            d = np.random.randn(self.dim)

            obj1 = np.max(self.vertices.dot(d))

            # Solve the second equation by cvxpy
            x = cvx.Variable(self.dim)
            constraint = [self.A * x <= self.b]
            objective = cvx.Maximize(d * x)
            problem = cvx.Problem(objective, constraint)
            problem.solve(solver='MOSEK')
            if problem.status != 'optimal' or np.abs(problem.value - obj1) >= eps:
                return False, "Random projection failed! \n     {:f} != {:f}".format(obj1, problem.value)
        return True, "Validated successfull with {:d} vectors".format(N_vectors)

    def _compute_volume_qhull(self):
        """ Compute volume with external executable qhull.

        Note: require qhull-bin installed, and .temp folder created.

        Returns
        -------
        vol: float
            Volume of the convex hull.

        """
        input_file = '/home/hung/.temp/input.qhull'
        output_file = '/home/hung/.temp/output.qhull'
        np.savetxt(input_file, self.vertices)
        line_prepender(input_file, "{:d}\n{:d}".format(
            self.vertices.shape[1], self.vertices.shape[0]))  # Add the dof
        os.system("qconvex FA TI {:} TO {:}".format(input_file, output_file))
        with open(output_file, 'r') as f:
            content = f.readlines()
        vol = float(filter(lambda l: "volume" in l, content)[0].split(':')[1])
        return vol

    def _compute_volume_delaunay(self):
        dt = scipy.spatial.Delaunay(self.vertices)
        test = dt.points[dt.simplices]
        vol = np.sum(map(compute_volume_simplex, test))
        return vol

    def _compute_volume_monte(self, n_samples=1000):
        d = np.max(np.abs(self.get_vertices()))
        random_pts = sample_uniform(n_samples, d, 6)

        density = (2 * d) ** 6 / n_samples
        n_pt_in = 0
        for pt in random_pts:
            if np.all(self.A.dot(pt) <= self.b):
                n_pt_in += 1
        return float(n_pt_in) * density


