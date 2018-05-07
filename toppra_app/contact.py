import numpy as np
from .utils import Ad
from .rave_fixed_frame import RaveRobotFixedFrame


class Contact(RaveRobotFixedFrame):
    """A class representing the contact between the robot an external object.

    Note that the frame {contact} is not necessarily placed at the
    center of the physical contact. It can be anywhere, even floating
    in empty space. This frame is only use to specify the physical
    meaning of F_local and g_local.

    Concretely,

               F_{local} w_{contact} \leq g_{local}

    Parameters
    ----------
    robot: openravepy.Robot
    link_name: string
        Name of the link, or the end-effect, at which the contact is
        attached to.
    T_link_contact: (4,4)array
        Transformation of {contact} in {link}.
    F_local: (m,6)array
        Constraint coefficient. See above.
    g_local: (m,)array
        Constraint coefficient. See above.
    """
    def __init__(self, robot, attached_name, T_attached_frame, F_local, g_local, dofindices=None):
        super(Contact, self).__init__(robot, attached_name, T_attached_frame, dofindices)
        self.F_local = F_local
        self.g_local = g_local

    def get_constraint_coeffs_local(self):
        return self.F_local, self.g_local

    def compute_constraint_coeffs(self, q, T_world_frame, dofindices=None):
        """ Return the global constraint .. math:: F w \leq g.

        Parameters
        ----------
        q: array
            Joint position.
        T_world_frame: array
            The 4x4 matrix homogeneous coordinate of {frame} in {world}.
        dofindices: list of int
            Selected Indices.

        Returns
        -------
        F_global: array
             Constraint coefficient.
        g_global: array
             Constraint coefficient.

        """
        T_world_contact = self.compute_frame_transform(q, dofindices=dofindices)
        T_frame_contact = T_world_frame.inv().dot(T_world_contact)
        F_frame = self.F_local.dot(Ad(T_frame_contact).T)
        g_frame = np.copy(self.g_local)
        return F_frame, g_frame
