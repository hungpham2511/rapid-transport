"""
This script is similar to test1.py, but utilizes the new toppra api.

"""
import numpy as np
import openravepy as orpy
import toppra as ta
import pytest

import toppra_app


@pytest.fixture(scope="module")
def setup():
    env = orpy.Environment()
    # env.Load('../models/denso_ft_gripper_with_base.robot.xml')
    env.Load("data/lab1.env.xml")
    robot = env.GetRobots()[0]
    active_indices = range(6)
    robot.SetActiveDOFs(active_indices)
    ft_name = 'wam6'
    yield robot, ft_name


@pytest.mark.parametrize("utest, xtest", [
    [0.5, 0.6], [0, 1.0], [1.0, 0], [0, 0]])
@pytest.mark.parametrize("T_eo", [
    np.array([[0, 1, 0, 0.0e-3],
              [0, 0, 1, -0.0425 + 25.2e-3 / 2],
              [1, 0, 0, 0.16796 - 25.2e-3 / 2],
              [0, 0, 0, 1]]), np.eye(4)])
@pytest.mark.parametrize("si", [0.5, 1])
def test_basic_contact_obj_coincides(setup, si, T_eo, utest, xtest):
    """Test a simple case where two frames {contact} and {object}
    coincide.

    """
    robot, ft_name = setup
    # Inertial properties
    m = 0.1
    lx = 25.2e-3
    ly = 62.9e-3
    lz = 25.2e-3
    I_o = m / 12 * np.diag([ly ** 2 + lz ** 2, lx ** 2 + lz ** 2, lx ** 2 + ly ** 2])
    solid_object = toppra_app.SolidObject(robot, ft_name, T_eo, m, I_o)

    # Contact properties:   -1 <= w <= 1
    F = np.vstack((np.eye(6), -np.eye(6)))
    g = np.hstack((np.ones(6), np.ones(6)))
    contact = toppra_app.Contact(robot, ft_name, T_eo, F, g)

    pc_object_trans = toppra_app.create_object_transporation_constraint(contact, solid_object)

    # TOPPRA parameters
    waypoints = np.array([
        [-0.5, 0.78, 0.78, 0, 0, -0.2],
        [-0.2, 0.78, 0.78, 0, 0.2, -0.3],
        [0.2, 0.78, 0.8, 0, 0., 0.4],
        [0.5, 0.78, 0.78, 0, 0, 0]])

    path = ta.SplineInterpolator(np.linspace(0, 1, 4), waypoints)
    params = pc_object_trans.compute_constraint_params(path, [si])[:5]
    ai, bi, ci, Fi, gi = zip(*params)[0]
    # Correct coefficient
    q_ = path.eval(si)
    qd_ = path.evald(si) * np.sqrt(xtest)
    qdd_ = path.evaldd(si) * xtest + path.evald(si) * utest

    T_contact = contact.compute_frame_transform(q_)
    w_contact = solid_object.compute_inverse_dyn(q_, qd_, qdd_, T_contact)

    np.testing.assert_allclose(F, Fi)
    np.testing.assert_allclose(g, gi)
    np.testing.assert_allclose(ai * utest + bi * xtest + ci, w_contact)
    np.testing.assert_allclose(Fi.dot(ai * utest + bi * xtest + ci) - gi, Fi.dot(w_contact) - gi)



