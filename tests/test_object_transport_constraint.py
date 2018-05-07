"""
This script is similar to test1.py, but utilizes the new toppra api.

"""
import numpy as np
import openravepy as orpy
import time
import toppra as ta
import matplotlib.pyplot as plt
import pytest

import toppra_app
import coloredlogs

coloredlogs.install(level='INFO')
np.set_printoptions(3)


def create_object_transporation_constraint(contact, solid_object):
    """

    Parameters
    ----------
    contact: Contact
    solid_object: SolidObject

    Returns
    -------
    constraint:

    """
    def inv_dyn(q, qd, qdd):
        T_contact = contact.compute_frame_transform(q)
        wrench_contact = solid_object.compute_inverse_dyn(q, qd, qdd, T_contact)
        return wrench_contact

    def cnst_F(q):
        return contact.get_constraint_coeffs_local()[0]

    def cnst_g(q):
        return contact.get_constraint_coeffs_local()[1]

    constraint = ta.constraint.CanonicalLinearSecondOrderConstraint(inv_dyn, cnst_F, cnst_g)
    return constraint


@pytest.mark.skip(reason="unable to test for now")
def test_overall():
    env = orpy.Environment()
    env.Load('../models/denso_ft_gripper_with_base.robot.xml')
    robot = env.GetRobots()[0]
    manip = robot.SetActiveManipulator('denso_ft_sensor')
    arm_indices = manip.GetArmIndices()
    ft_name = 'link6'
    env.SetViewer('qtosg')

    # Problem parameter
    g_w = np.array([0, 0, -9.8])
    # Pose of the object's frame in end-effector frame
    T_eo = np.array([[0, 1, 0, 0.0e-3],
                     [0, 0, 1, -0.0425 + 25.2e-3 / 2],
                     [1, 0, 0, 0.16796 - 25.2e-3 / 2],
                     [0, 0, 0, 1]])

    R_eo = T_eo[:3, :3]
    p_eo = T_eo[:3, 3]
    # Inertial properties
    m = 0.1
    lx = 25.2e-3
    ly = 62.9e-3
    lz = 25.2e-3
    I_o = m / 12 * np.diag([ly ** 2 + lz ** 2, lx ** 2 + lz ** 2, lx ** 2 + ly ** 2])

    solid_object = toppra_app.SolidObject(robot, manip.GetName(), T_eo, m, I_o, dofindices=arm_indices)

    # Contact properties
    file = np.load("../data/alum_block_rubber_contact_contact.npz")
    A = file['A']
    b = file['b']
    contact = toppra_app.Contact(robot, manip.GetName(), T_eo, A, b, dofindices=arm_indices)

    pc_object_trans = create_object_transporation_constraint(contact, solid_object)

    # TOPPRA parameters
    N = 50
    velocity_safety_factor = 0.65
    acceleration_safety_factor = 0.65

    waypoints = np.array([
        [-0.5, 0.78, 0.78, 0, 0, -0.2],
        [-0.2, 0.78, 0.78, 0, 0.2, -0.3],
        [0.2, 0.78, 0.8, 0, 0., 0.4],
        [0.5, 0.78, 0.78, 0, 0, 0]])

    # print "Preview the waypoints"
    # for i, q in enumerate(waypoints):
    #     robot.SetDOFValues(q, manip.GetArmIndices())
    #     raw_input("Waypoint no. {:d}. [Enter] to continue!".format(i))
    path = ta.SplineInterpolator(np.linspace(0, 1, 4), waypoints)
    ss = np.linspace(0, 1, N + 1)

    params = pc_object_trans.compute_constraint_params(path, ss)
    a, b, c, F, g, _, _ = params

    a_vec_totest = [F[i].dot(a[i]) for i in range(N + 1)]
    b_vec_totest = [F[i].dot(b[i]) for i in range(N + 1)]
    c_vec_totest = [F[i].dot(c[i]) - g[i] for i in range(N + 1)]

    # Correct coefficient
    f_ = np.load("object_transport_test.npz")
    a_vec = f_['a_vec']
    b_vec = f_['b_vec']
    c_vec = f_['c_vec']
    theta1 = f_['theta1']
    theta2 = f_['theta2']
    theta3 = f_['theta3']

    # test
    np.testing.assert_allclose(theta1, a, atol=1e-8)
    np.testing.assert_allclose(theta2, b, atol=1e-8)
    np.testing.assert_allclose(theta3, c, atol=1e-8)

    np.testing.assert_allclose(a_vec, a_vec_totest, atol=1e-8)
    np.testing.assert_allclose(b_vec, b_vec_totest, atol=1e-8)
    np.testing.assert_allclose(c_vec, c_vec_totest, atol=1e-8)


