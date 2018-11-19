import toppra
from toppra.constraint import CanonicalLinearConstraint, DiscretizationType
import numpy as np


class ObjectTransportationConstraint(CanonicalLinearConstraint):
    """ Object transporation constraint.

    Parameters
    ----------
    inv_dyn : func: (R^n, R^n, R^n) -> R^n
        Inverse dynamic function.
    cnst_F : array
    cnst_g : array
    dof : , optional
        FIXME
    discretization_scheme : , optional
        FIXME

    Returns
    -------
    out :
        FIXME   
    
    """
    def __init__(self, inv_dyn, cnst_F, cnst_g, dof=None, discretization_scheme=1):
        super(ObjectTransportationConstraint, self).__init__()
        self.set_discretization_type(discretization_scheme)
        self.inv_dyn = inv_dyn
        self.cnst_F = cnst_F
        self.cnst_g = cnst_g
        self.dof = dof
        self._format_string = "    Kind: Generalized Second-order constraint\n"
        self._format_string = "    Dimension:\n"
        self._format_string += "        F in R^({:d}, {:d})\n".format(*cnst_F.shape)
        self.identical = True

    def compute_constraint_params(self, path, gridpoints, scaling):
        if path.get_dof() != self.get_dof():
            raise ValueError("Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                self.get_dof(), path.get_dof()
            ))
        if scaling != 1:
            raise(NotImplementedError("Scaling functionality not implemented."))
        v_zero = np.zeros(path.get_dof())
        p = path.eval(gridpoints)
        ps = path.evald(gridpoints)
        pss = path.evaldd(gridpoints)

        F = np.array(self.cnst_F)
        g = np.array(self.cnst_g)
        c = np.array(
            map(lambda p_: self.inv_dyn(p_, v_zero, v_zero), p)
        )
        a = np.array(
            map(lambda (p_, ps_): self.inv_dyn(p_, v_zero, ps_), zip(p, ps))
        ) - c
        b = np.array(
            map(lambda (p_, ps_, pss_): self.inv_dyn(p_, ps_, pss_), zip(p, ps, pss))
        ) - c

        if self.discretization_type == 0 or self.discretization_type == DiscretizationType.Collocation:
            return a, b, c, F, g, None, None
        elif self.discretization_type == 1 or self.discretization_type == DiscretizationType.Interpolation:
            return toppra.constraint.canlinear_colloc_to_interpolate(a, b, c, F, g, None, None, gridpoints, identical=self.identical)
        else:
            raise(NotImplementedError("Other form of discretization not supported!"))


def create_object_transporation_constraint(contact, solid_object, discretization_scheme=0):
    """Create a TOPP constraint from a `Contact` and a `SolidObject`.

    Each pair of (contact, object) forms a Second-Order Canonical
    Linear constraint. This kind of constraint can be handled by
    `toppra`.

    Examples include:

    - {contact} is with the ground, {object} is the full robot.

    - {contact} is between the robot and an object, {object} is the
      object to transport.

    Here note that the constraint is formed at the origin of frame
    {contact}. The reason for this choice is that matrices F and g can
    be large, and hence are expensive to transform.

    Parameters
    ----------
    contact: Contact
    solid_object: SolidObject
    discretization_scheme: bool

    Returns
    -------
    constraint: toppra.SecondOrderConstraint

    """
    def inv_dyn(q, qd, qdd):
        T_contact = contact.compute_frame_transform(q)
        wrench_contact = solid_object.compute_inverse_dyn(q, qd, qdd, T_contact)
        return wrench_contact

    # def cnst_F(q):
    #     return contact.get_constraint_coeffs_local()[0]

    # def cnst_g(q):
    #     return contact.get_constraint_coeffs_local()[1]

    # constraint = toppra.constraint.CanonicalLinearSecondOrderConstraint(
    #     inv_dyn, cnst_F, cnst_g, dof=solid_object.get_robot().GetActiveDOF())

    F, g = contact.get_constraint_coeffs_local()
    constraint = ObjectTransportationConstraint(
        inv_dyn, F, g, dof=solid_object.get_robot().GetActiveDOF(), discretization_scheme=discretization_scheme)

    return constraint


