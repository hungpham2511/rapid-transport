from utils import compute_Jacobians, compute_Hessians, transform_inv
import numpy as np


class RaveRobotFixedFrame(object):
    """ Base class for objects that attached to a link of the robot.

    Parameters
    ----------
    robot: openravepy.Robot
    attached_name: str
        Name of an manipulator, or a link on the robot.
    T_attached_frame: array
        Transform from the local frame of attached to the object frame.
    dofindices: int list, optional
        List of joint indices. If None select all joints.
    """

    def __init__(self, robot, attached_name, T_attached_frame, dofindices=None):
        self.robot = robot
        link = robot.GetLink(attached_name)
        manip = robot.GetManipulator(attached_name)
        if link is None and manip is not None:
            self.link_name = manip.GetEndEffector().GetName()
            T_world_ee = manip.GetTransform()
            T_world_frame = T_world_ee.dot(T_attached_frame)
            T_world_link = manip.GetEndEffector().GetTransform()
            self.T_link_contact = transform_inv(T_world_link).dot(T_world_frame)
        elif link is not None and manip is None:
            self.link_name = attached_name
            self.T_link_contact = T_attached_frame
        else:
            raise ValueError, "[{:}] is not a link name or manipulator name.".format(attached_name)

        self.dofindices = dofindices

    def compute_frame_transform(self, q, dofindices=None):
        """ The frame coordinate in the world frame.

        Parameters
        ----------
        q: array
            Joint position.
        dofindices: list of int
            Selected Indices.

        Returns
        -------
        T_world_contact: array
            Shape (4,4). Transform of the contact frame in the world frame when the robot
            has joint position q. Note that dofindices should contain at least all joints
            that might affect the transformation.
        """
        if dofindices is None:
            dofindices = self.dofindices
        with self.robot:
            self.robot.SetDOFValues(q, dofindices)
            link = self.robot.GetLink(self.link_name)
            T_world_link = link.GetTransform()
            T_world_contact = T_world_link.dot(self.T_link_contact)
        return T_world_contact

    def compute_kinematics_global(self, q, qd, qdd, dofindices=None):
        """ Return the kinematic quantities in the world frame.

        Returns
        -------
        a: array
            Linear acceleration.
        v: array
            Linear velocity.
        alpha: array
            Angular acceleration.
        omega: array
            Angular velocity.
        """
        if dofindices is None:
            dofindices = self.dofindices
        p_endlink = self.T_link_contact[:3, 3]
        J_rot, J_tran = compute_Jacobians(self.robot, self.link_name, q_cur=q, pos_endlink=p_endlink)
        H_rot, H_tran = compute_Hessians(self.robot, self.link_name, q_cur=q, pos_endlink=p_endlink)

        v = J_tran.dot(qd)
        a = J_tran.dot(qdd) + np.dot(qd, np.dot(H_tran, qd))
        omega = J_rot.dot(qd)
        alpha = J_rot.dot(qdd) + np.dot(qd, np.dot(H_rot, qd))

        return a, v, alpha, omega

    def compute_kinematics_local(self, q, qd, qdd, dofindices=None):
        """ Return the kinematic quantities in the local frame.

        The kinematic quantities, angular and linear velocity and acceleration,
        are vectors defined in an inertia frame. The actual coordinates, however, are
        transformed to different frame.

        Returns
        -------
        a: array
            Linear acceleration.
        v: array
            Linear velocity.
        alpha: array
            Angular acceleration.
        omega: array
            Angular velocity.
        """
        if dofindices is None:
            dofindices = self.dofindices
        a, v, alpha, omega = self.compute_kinematics_global(q, qd, qdd)
        R_world_contact = self.compute_frame_transform(q, dofindices)[:3, :3]
        results = map(lambda x: R_world_contact.T.dot(x), (a, v, alpha, omega))
        return results

