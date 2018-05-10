import numpy as np
from .utils import Ad, expand_and_join
from .rave_fixed_frame import RaveRobotFixedFrame
from .profile_loading import Database


class Contact(RaveRobotFixedFrame):
    """A class representing the contact between the robot an external
    object.

    Note that the frame {contact} is not necessarily placed at the
    center of the physical contact. It can be anywhere, even floating
    in empty space. This frame is only use to specify the physical
    meaning of F_local and g_local.

    Concretely,

               F_{local} w_{contact} \leq g_{local}

    This object directly corresponds to the contact profiles in
    contacts.yaml. This is not true for object profiles in
    objects.yaml. The key difference is that in the latter, the
    profile does not specify which link, or manipulator that the
    object is attached to.

    A contact can contain non-empty `raw_data` field. This contact is
    called raw. Often, a contact of this kind needs to be simplifed
    with `ContactSimplifier` before it can be used.

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
    dofindices: THIS IS DEPRECEATED
    profile_id: str
        Id of the profile that this contact is loaded from.
    raw_data: list
        List of raw data file names.

    """
    def __init__(self, robot, link_name, T_link_contact, F_local, g_local, dofindices=None, profile_id="", raw_data=[]):
        super(Contact, self).__init__(robot, link_name, T_link_contact, dofindices)
        self.F_local = F_local
        self.g_local = g_local
        self._profile_id = profile_id
        self._raw_data = raw_data

    @staticmethod
    def init_from_profile_id(robot, profile_id):
        """ Initialization from a contact profile id.

        Parameters
        ----------
        robot: openravepy.Robot
            The robot at which this contact attaches to.
        profile_id: str
        """
        db = Database()
        contact_profile = db.retrieve_profile(profile_id, "contact")
        try:
            with np.load(expand_and_join(db.get_contact_data_dir(), contact_profile['constraint_coeffs_file'])) as f:
                F = f['A']
                g = f['b']
        except KeyError:
            F = None
            g = None

        try:
            raw_data = contact_profile['raw_data']
        except KeyError:
            raw_data = []

        T_link_contact = np.eye(4)
        T_link_contact[:3, 3] = contact_profile['position']
        T_link_contact[:3, :3] = contact_profile['orientation']
        return Contact(robot, contact_profile['attached_to_manipulator'],
                       T_link_contact, F, g, profile_id=profile_id, raw_data=raw_data)

    def clone(self):
        """ Return a cloned contact object.
        """
        return Contact(self.robot, self._attached_name, self.T_link_contact,
                       self.F_local, self.g_local, profile_id=self._profile_id)

    def get_raw_data(self):
        return self._raw_data

    def get_profile_id(self):
        return self._profile_id

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
