import numpy as np
from math import factorial
from toppra.constraint import CanonicalLinearSecondOrderConstraint
import toppra
import os
import matplotlib.pyplot as plt
import logging
import coloredlogs

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    "Create a stream handler which outputs msg to console"
    coloredlogs.install(logger=logging.getLogger("transport"), level=level,
                        fmt="%(levelname)s %(asctime)s (%(name)s) [%(funcName)s: %(lineno)d] %(message)s",
                        datefmt="%H:%M:%S", milliseconds=True)


def preview_plot(args, dur=3):
    """Preview data tuples given in args.

    Each argument in `args` is (points, marker, params_dict).

    Each `points` is a (N,6) float array. The first three dimensions
    are for torque and the last three dimensions are for force.

    The simplest parameter dictionary is {}.

    Parameters
    ----------
    args: list of ((N,d)array, marker, size)
    """
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].set_title('tau_x vs tau_z')
    axs[0, 1].set_title('tau_y vs tau_z')
    axs[0, 2].set_title('tau_x vs tau_y')
    axs[1, 0].set_title('f_x vs f_z')
    axs[1, 1].set_title('f_y vs f_z')
    axs[1, 2].set_title('f_x vs f_y')
    for ws_all, fmt, params in args:
        axs[0, 0].plot(ws_all[:, 0], ws_all[:, 2], **params)
        axs[0, 1].plot(ws_all[:, 1], ws_all[:, 2], **params)
        axs[0, 2].plot(ws_all[:, 0], ws_all[:, 1], **params)
        axs[1, 0].plot(ws_all[:, 3], ws_all[:, 5], **params)
        axs[1, 1].plot(ws_all[:, 4], ws_all[:, 5], **params)
        axs[1, 2].plot(ws_all[:, 3], ws_all[:, 4], **params)

    # Setup close figure callback
    logger.debug("Showing figure, closing in {:.3f} seconds".format(dur))
    timer = fig.canvas.new_timer(interval=int(1000 * dur))
    timer.add_callback(lambda: plt.close(fig))
    timer.start()
    plt.tight_layout()
    plt.show()


def expand_and_join(path1, path2):
    return os.path.expanduser(os.path.join(path1, path2))


def compute_area_ndim(vertices):
    """ Area of the simplex form by vertices.

    Parameters
    ----------
    vertices: (n+1,n)array
        Array of vertices

    Returns
    -------
    out: float
        Area of the simplex.
    """
    vertices = np.array(vertices)
    dof = vertices.shape[1]
    assert vertices.shape[0] == dof

    col = (vertices[1:] - vertices[0]).T
    Q, R = np.linalg.qr(col)
    return np.abs(np.linalg.det(R)) / factorial(dof - 1)


def create_second_order_constraint(contact, solid_object):
    """

    Parameters
    ----------
    contact: Contact
    solid_object: ArticulatedBody

    Returns
    -------
    constraint:

    """
    def inv_dyn(q, qd, qdd):
        T_contact = contact.compute_frame_transform(q)
        wrench_contact = solid_object.compute_inverse_dyn(q, qd, qdd, T_contact)
        return wrench_contact

    def cnst_F(q):
        return contact.get_constraint_coeffs_local()[0]

    def cnst_g(q):
        return contact.get_constraint_coeffs_local()[1]

    constraint = CanonicalLinearSecondOrderConstraint(inv_dyn, cnst_F, cnst_g)
    return constraint


def line_prepender(input_file, line):
    """ Prepend line to input_file.

    Parameters
    ----------
    input_file: string
        Path of the input_file.
    line: string
        Line to prepend.

    """
    with open(input_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def compute_volume_simplex(pts):
    dof = pts.shape[1]
    mat = pts[1:] - pts[0]
    return np.abs(1.0 / factorial(dof) * np.linalg.det(mat))


def generate_random_unit_vectors(dof, n):
    rand_vecs = []
    for i in range(n):
        vec = np.random.randn(dof)
        rand_vecs.append(vec / np.linalg.norm(vec))
    return rand_vecs


def sample_uniform(n_pts, d, dof):
    """ Sample n_pts within the box [-d, d]^dof

    Parameters
    ----------
    n_pts
    d
    dof

    Returns
    -------
    pts: (n_pts, dof)array
    """
    pts_unit = np.random.rand(n_pts, dof)
    pts_rand = d * (pts_unit - 0.5)
    return pts_rand


def load_data_ati_log(file_name):
    """ Load data from text files.

    The return wrenches are the wrenches that act on the object by the robot.
    This is different from the raw data produce by the FT sensor, which contains
    measurements of the wrenches that act on the FT sensor by the object.

    Parameters
    ----------
    file_name: string
        Name of the data file.

    Returns
    -------
    out : (-1,6)array
        List of wrenches in (torque,force).

    """
    print "Load data.."
    COUNTS = 1e6  # Number of ticks for unit torque/force
    with open(file_name) as f:
        content = f.readlines()
    content = content[10:]
    ws_e = []
    for line in content:
        fx, fy, fz, tx, ty, tz = [-float(s[1:-1]) / COUNTS for s in line.split(",")[3:9]]
        ws_e.append([tx, ty, tz, fx, fy, fz])
    ws_e = np.array(ws_e)
    return ws_e

def skew(p):
    mat = np.array(([0, -p[2], p[1]],
                    [p[2], 0, -p[0]],
                    [-p[1], p[0], 0]))
    return mat


def transform_inv(T):
    return np.linalg.inv(T)


def Ad(T):
    """ The Ad operator. See (Park and Lynch)

    Parameters
    ----------
    T

    Returns
    -------

    """
    mat = np.zeros((6, 6))
    mat[:3, :3] = T[:3, :3]
    mat[3:, 3:] = T[:3, :3]
    mat[3:, :3] = skew(T[:3, 3]).dot(T[:3, :3])
    return mat


def compute_Hessians(robot, endlink_name, q_cur, pos_world=None, pos_endlink=None):
    """ Compute the Hessians of the robot with respect to its active DOF.

    Parameters
    ----------
    robot: openravepy.Robot
    endlink_name: str
        Name of the end-effector link, or the link that the object is attached to.
    q_cur: ndarray
        Joint pos_endlink to compute those Jacobians at.
    pos_world: ndarray
        Position of the object's com in the world frame.
    pos_endlink: ndarray
        Position of the object's com in the end-effector link.

    Returns
    -------
    H_rot: ndarray
    H_trans: ndarray
    """
    with robot:
        active_indices = robot.GetActiveDOFIndices()
        robot.SetDOFValues(q_cur, active_indices)
        link = robot.GetLink(endlink_name)
        if pos_world is None:
            T_world_link = link.GetTransform()
            pos_world = T_world_link.dot(np.r_[pos_endlink, 1])[:3]
        H_trans = robot.ComputeHessianTranslation(link.GetIndex(), pos_world, active_indices)
        H_rot = robot.ComputeHessianAxisAngle(link.GetIndex(), active_indices)
    return H_rot, H_trans


def compute_Jacobians(robot, endlink_name, q_cur, pos_world=None, pos_endlink=None):
    """ Compute the Jacobians with respect to robot's active DOF.

    Parameters
    ----------
    robot: openravepy.Robot
    endlink_name: str
        Name of the end-effector link, or the link that the object is attached to.
    q_cur: ndarray
        Joint pos_endlink to compute those Jacobians at.
    pos_world: ndarray
        Position of the object's com in the world frame.
    pos_endlink: ndarray
        Position of the object's com in the end-effector link.

    Returns
    -------
    J_rot: ndarray
    J_trans: ndarray
    """
    with robot:
        active_indices = robot.GetActiveDOFIndices()
        robot.SetDOFValues(q_cur, active_indices)
        link = robot.GetLink(endlink_name)
        if pos_world is None:
            T_world_link = link.GetTransform()
            pos_world = T_world_link.dot(np.r_[pos_endlink, 1])[:3]
        J_trans = robot.ComputeJacobianTranslation(link.GetIndex(), pos_world, active_indices)
        J_rot = robot.ComputeJacobianAxisAngle(link.GetIndex(), active_indices)
    return J_rot, J_trans

