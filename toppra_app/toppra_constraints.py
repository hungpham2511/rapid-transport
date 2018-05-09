import toppra


def create_object_transporation_constraint(contact, solid_object):
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

    Returns
    -------
    constraint: toppra.SecondOrderConstraint

    """
    def inv_dyn(q, qd, qdd):
        T_contact = contact.compute_frame_transform(q)
        wrench_contact = solid_object.compute_inverse_dyn(q, qd, qdd, T_contact)
        return wrench_contact

    def cnst_F(q):
        return contact.get_constraint_coeffs_local()[0]

    def cnst_g(q):
        return contact.get_constraint_coeffs_local()[1]

    constraint = toppra.constraint.CanonicalLinearSecondOrderConstraint(
        inv_dyn, cnst_F, cnst_g, dof=solid_object.get_robot().GetActiveDOF())
    return constraint


