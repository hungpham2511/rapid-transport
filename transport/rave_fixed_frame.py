from utils import compute_Jacobians, compute_Hessians, transform_inv
import numpy as np


class RaveRobotFixedFrame(object):
    """The base class for objects or frames attached to a link or a
    manipulator of an OpenRAVE robot.

    Parameters
    ----------
    robot: openravepy.Robot
    attached_name: str
        Name of a manipulator or a link.
    T_attached_frame: array
        Transform from the local frame of the attached link/manipulator to frame.
    dofindices: int list, optional
        List of joint indices. If None select all joints.
    """
    def __init__(self, robot, attached_name, T_attached_frame, dofindices=None):
        self.robot = robot
        link = robot.GetLink(attached_name)
        if link is not None:
            self._link = link
            self._T_link_frame = np.array(T_attached_frame)
        else:
            manip = robot.GetManipulator(attached_name)
            if manip is None:
                raise ValueError("Unable to find link or manipulator: {:}".format(attached_name))
            self._link = manip.GetEndEffector()
            self._T_link_frame = np.dot(manip.GetLocalToolTransform(), T_attached_frame)
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
            self.robot.SetActiveDOFValues(q)
            T_world_link = self._link.GetTransform()
            T_world_frame = T_world_link.dot(self._T_link_frame)
        return T_world_frame

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
        self.robot.SetActiveDOFValues(q)
        self.robot.SetActiveDOFVelocities(qd)
        linkvel = self.robot.GetLinkVelocities()[self._link.GetIndex()]
        qdd_full = np.zeros(self.robot.GetDOF())
        qdd_full[:qdd.shape[0]] = qdd
        linkaccel = self.robot.GetLinkAccelerations(qdd_full)[self._link.GetIndex()]
        vtranslinkframe = self._link.GetTransform()[:3, :3].dot(self._T_link_frame[:3, 3])

        omega = linkvel[3:]
        alpha = linkaccel[3:]
        v = linkvel[:3] + np.cross(omega, vtranslinkframe)
        a = linkaccel[:3] + np.cross(alpha, vtranslinkframe) + np.cross(omega, np.cross(omega, vtranslinkframe))
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

