
class ArticulatedBody(object):
    """ Base class for objects that compute wrenches.

    Two inverse dynamics methods are implemented.
    The first compute the wrench in the class' local frame.
    The second computes the wrench in an arbitrary frame.
    """
    def __init__(self):
        pass

    def compute_inverse_dyn_local(self, q, qd, qdd, dofindices=None):
        """ Compute the required wrench in the local frame {object}.

        Parameters
        ----------
        q: array
        qd: array
        qdd: array
        dofindices: list of int, optional

        Returns
        -------
        w: array
            The net interacting wrench in frame {object}.

        """
        raise NotImplementedError

    def compute_inverse_dyn(self, q, qd, qdd, T_world_frame, dofindices=None):
        """ Inverse dynamics wrench in {frame}.

        Parameters
        ----------
        q: (dof,)array
        qd: (dof,)array
        qdd: (dof,)array
        T_world_frame: (4,4)array
            Transform of {frame} in {world}.
        dofindices: list of int, optional
            List of active joint indices.

        Returns
        -------
        w: array
            The net interacting wrench in frame {object}.
        """
        raise NotImplementedError
