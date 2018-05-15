from .articulatedbody import ArticulatedBody
import numpy as np
from utils import transform_inv, Ad


class BaseWrenchManipulator(ArticulatedBody):
    """ A class for computing the wrench required at the base of a robot.

    Parameters
    ----------
    robot: openravepy.Robot
    dofindices: [int]
        List of active joint indices.
    """
    def __init__(self, robot, dofindices=None, g_world=np.array([0, 0, -9.8])):
        self.robot = robot
        self.g_world = g_world
        self.base_link = robot.GetLinks()[0]
        self.dofindices = dofindices
        self.T_base_world = transform_inv(self.base_link.GetTransform())

    def compute_inverse_dyn_local(self, q, qd, qdd, dofindices=None):
        """ Inverse dynamics at the origin of the base frame.

        Parameters
        ----------
        q: (dof,)array
        qd: (dof,)array
        qdd: (dof,)array
        dofindices: [int]

        Returns
        -------
        out: (6,)array
            Wrench.
        """
        N = len(self.robot.GetLinks())
        self.robot.SetDOFValues(q)

        wrench_base = np.zeros(6)
        for i in range(N):
            link = self.robot.GetLinks()[i]
            p = link.GetGlobalCOM()
            J_trans = self.robot.ComputeJacobianTranslation(i, p)
            H_trans = self.robot.ComputeHessianTranslation(i, p)
            J_rot = self.robot.ComputeJacobianAxisAngle(i)
            H_rot = self.robot.ComputeHessianAxisAngle(i)

            v = np.dot(J_trans, qd)
            vd = (np.dot(J_trans, qdd) + np.dot(qd, np.dot(H_trans, qd)))
            w = np.dot(J_rot, qd)
            wd = np.dot(J_rot, qdd) + np.dot(qd, np.dot(H_rot, qd))

            I = link.GetGlobalInertia()
            m = link.GetMass()

            wrench_link = np.zeros(6)
            wrench_link[:3] = I.dot(wd) + np.cross(w, I.dot(w))
            wrench_link[3:] = m * vd - m * self.g_world

            T_world_link = np.eye(4)
            T_world_link[:3, 3] = link.GetTransform()[:3, 3]
            T_base_link = self.T_base_world.dot(T_world_link)
            wrench_link_base = Ad(transform_inv(T_base_link)).T.dot(wrench_link)
            wrench_base = wrench_base + wrench_link_base
        return wrench_base

    def compute_inverse_dyn(self, q, qd, qdd, T_world_frame, dofindices=None):
        wrench_base = self.compute_inverse_dyn_local(q, qd, qdd, dofindices)
        T_base_frame = self.T_base_world.dot(T_world_frame)
        return Ad(T_base_frame).T.dot(wrench_base)

