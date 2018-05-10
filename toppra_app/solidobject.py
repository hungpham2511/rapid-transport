from .rave_fixed_frame import RaveRobotFixedFrame
from .articulatedbody import ArticulatedBody
from .profile_loading import Database
from .contact import Contact
import numpy as np
from .utils import Ad, transform_inv, expand_and_join
import logging
logger = logging.getLogger(__name__)


class SolidObject(RaveRobotFixedFrame, ArticulatedBody):
    """ A class represents a the dynamics of a solid object.

    The geometry of an object, such as its mesh, is not used.

    Here the frame {object} is a frame whose origin coincides with the robot's com.

    Parameters
    ----------
    robot: openrave.Robot
    _attached_name: str
        Name of the link the object is attached to.
    T_link_object: (4,4)array
        Transformation matrix from {link} to {object}.
    m: float
        Weight.
    I_local: (3,3)array
        Local moment of inertia.
    dofindices: list of int, optional
        List of active indices. This parameter is deprecated. Now use the active DOFs of
        the robot by default.
    contact: optional
    profile
    """
    def __init__(self, robot, attached_name, T_link_object, m, I_local, dofindices=None, contact=None, profile="", name="", rave_model_path="", T_object_model=None):
        super(SolidObject, self).__init__(robot, attached_name, T_link_object, dofindices)
        self.m = m
        self.g_world = np.r_[0, 0, -9.8]
        self.I_local = np.array(I_local)
        self._profile = profile
        self._name = name
        self._contact = contact
        self._robot = robot
        self._env = robot.GetEnv()
        self._rave_model_path = rave_model_path
        self._T_object_model = T_object_model
        self._T_object_link = np.linalg.inv(T_link_object)
        self._T_link_object = np.array(T_link_object)

    @staticmethod
    def init_from_dict(robot, input_dict):
        """ Initialize from an input dictionary.

        Parameters
        ----------
        robot: openravepy.Robot
        input_dict: dict
        """
        assert input_dict['object_attach_to'] is not None
        try:
            contact = Contact.init_from_profile_id(robot, input_dict['contact_profile'])
        except KeyError:
            contact = None
        db = Database()
        object_profile = db.retrieve_profile(input_dict['object_profile'], "object")
        T_link_object = np.array(input_dict["T_link_object"], dtype=float)
        T_object_model = np.array(object_profile["T_object_model"], dtype=float)
        rave_model_path = expand_and_join(db.get_model_dir(), object_profile['rave_model'])
        solid_object = SolidObject(robot, input_dict['object_attach_to'], T_link_object, object_profile['mass'],
                                   object_profile['local_inertia'], contact=contact, profile=input_dict['object_profile'],
                                   name=input_dict['name'], rave_model_path=rave_model_path, T_object_model=T_object_model)
        return solid_object

    def get_profile(self):
        return self._profile

    def get_name(self):
        return self._name

    def get_contact(self):
        return self._contact

    def get_robot(self):
        return self._robot

    def get_T_object_link(self):
        return self._T_object_link

    def load_to_env(self, T_object=None, T_model=None):
        if not self._env.Load(self._rave_model_path):
            logger.warn("Unable to load model {:} to rave environment".format(self._rave_model_path))

        rave_body = self._env.GetBodies()[-1]
        rave_body.SetName(self._name)

        if T_object is not None:
            T_model = np.dot(T_object, self._T_object_model)
        rave_body.SetTransform(T_model)

        return rave_body

    def compute_inverse_dyn_local(self, q, qd, qdd, dofindices=None):
        """ The net wrench in the local frame {object} as the robot moves.

        Parameters
        ----------
        q: (d,)array
        qd: (d,)array
        qdd: (d,)array

        Returns
        -------
        w: (6,)array
            (torque,force) wrench acting at {object}'s origin.
        """
        a, v, alpha, omega = self.compute_kinematics_local(q, qd, qdd, dofindices)
        R_world_body = self.compute_frame_transform(q, dofindices)[:3, :3]
        g_body = R_world_body.T.dot(self.g_world)
        wrench = np.zeros(6)
        wrench[:3] = self.I_local.dot(alpha) + np.cross(omega, self.I_local.dot(omega))
        wrench[3:] = self.m * a - self.m * g_body
        return wrench

    def compute_inverse_dyn(self, q, qd, qdd, T_world_frame, dofindices=None):
        """ Net wrench the robot exerted on the object in frame {frame}.

        {frame}'s transformation in the world frame {world} is given by T_world_frame.

        Parameters
        ----------
        q
        qd
        qdd
        T_world_frame: (4,4)array
            Transformation of {frame} in the world frame.

        Returns
        -------
        w: (6,)array
            (torque,force) wrench acting at {frame}'s origin.
        """
        T_world_body = self.compute_frame_transform(q, dofindices)
        T_body_frame = transform_inv(T_world_body).dot(T_world_frame)
        wrench_body = self.compute_inverse_dyn_local(q, qd, qdd, dofindices)
        wrench_frame = Ad(T_body_frame).T.dot(wrench_body)
        return wrench_frame
