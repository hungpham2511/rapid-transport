from .rave_fixed_frame import RaveRobotFixedFrame
from .articulatedbody import ArticulatedBody
import numpy as np
from .utils import Ad, transform_inv


class SolidObject(RaveRobotFixedFrame, ArticulatedBody):
    """ A class represents a solid object.

    Parameters
    ----------
    robot: openrave.Robot
    link_name: str
        Name of the link the object is attached to.
    T_link_object: (4,4)array
        Transformation matrix.
    m: float
        Object weight.
    I_local: (3,3)array
        Local object inertia.
    dofindices: list of int, optional
        List of active indices.
    """
    def __init__(self, robot, attached_name, T_link_object, m, I_local, dofindices=None):
        super(SolidObject, self).__init__(robot, attached_name, T_link_object, dofindices)
        self.m = m
        self.g_world = np.r_[0, 0, -9.8]
        self.I_local = np.array(I_local)

    def compute_inverse_dyn_local(self, q, qd, qdd, dofindices=None):
        """ Net wrench in object local frame.

        Parameters
        ----------
        q
        qd
        qdd
        dofindices

        Returns
        -------

        """
        a, v, alpha, omega = self.compute_kinematics_local(q, qd, qdd, dofindices)
        R_world_body = self.compute_frame_transform(q, dofindices)[:3, :3]
        g_body = R_world_body.T.dot(self.g_world)
        wrench = np.zeros(6)
        wrench[:3] = self.I_local.dot(alpha) + np.cross(omega, self.I_local.dot(omega))
        wrench[3:] = self.m * a - self.m * g_body
        return wrench

    def compute_inverse_dyn(self, q, qd, qdd, T_world_frame, dofindices=None):
        """ Net wrench in {frame} defined by given transform matrix.

        Parameters
        ----------
        q
        qd
        qdd
        T_world_frame
        dofindices

        Returns
        -------

        """
        T_world_body = self.compute_frame_transform(q, dofindices)
        T_body_frame = transform_inv(T_world_body).dot(T_world_frame)
        wrench_body = self.compute_inverse_dyn_local(q, qd, qdd, dofindices)
        wrench_frame = Ad(T_body_frame).T.dot(wrench_body)
        return wrench_frame
