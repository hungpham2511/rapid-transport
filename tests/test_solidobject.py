import pytest
import openravepy as orpy
import toppra_app
from toppra_app.utils import compute_Hessians, compute_Jacobians
import numpy as np

@pytest.fixture()
def setup():
    env = orpy.Environment()
    env.Load('../models/denso_ft_gripper_with_base.robot.xml')
    robot = env.GetRobots()[0]
    manip = robot.SetActiveManipulator('denso_ft_sensor')
    arm_indices = manip.GetArmIndices()
    ft_name = 'link6'

    yield robot


def test_inv_dyn_frame_static(setup):
    """  robot static, com coincides with link origin
    """
    robot = setup
    T_eo = np.eye(4)
    m = 1
    I_o = np.eye(3)
    arm_indices = range(6)

    solid_object = toppra_app.SolidObject(robot, 'link6', T_eo, m, I_o, dofindices=arm_indices)
    link = robot.GetLink('link6')

    # Case 1: static and simple pose
    q_totest = [
        np.zeros(6), np.r_[1, 0, 1, 0, 1, 0]
    ]
    qd0 = np.zeros(6)
    qdd0 = np.zeros(6)

    for q0 in q_totest:
        with robot:
            robot.SetDOFValues(q0, arm_indices)
            T0 = np.eye(4)
            T0[:3, 3] = link.GetTransform()[:3, 3]

        w1 = solid_object.compute_inverse_dyn(q0, qd0, qdd0, T0)

        np.testing.assert_allclose(w1, np.r_[0, 0, 0, 0, 0, 9.8], atol=1e-8)
        assert np.allclose(w1, np.r_[0, 0, 0, 0, 0, 9.8])


def test_static_not_coincide(setup):
    robot = setup
    T_link_obj = np.array([[1, 0, 0, 0.4],
                           [0, 0, -1, 0.2],
                           [0, 1, 0, -0.2],
                           [0, 0, 0, 1]])
    m = 1
    I_o = np.eye(3)
    arm_indices = range(6)

    solid_object = toppra_app.SolidObject(robot, 'link6', T_link_obj, m, I_o, dofindices=arm_indices)
    link = robot.GetLink('link6')

    # Case 1: static and simple pose
    q_totest = [
        np.zeros(6), np.r_[1, 0, 1, 0, 1, 0]
    ]
    qd0 = np.zeros(6)
    qdd0 = np.zeros(6)

    for q0 in q_totest:
        with robot:
            robot.SetDOFValues(q0, arm_indices)
            T0 = np.eye(4)
            T0[:3, 3] = link.GetTransform().dot(T_link_obj)[:3, 3]

        w1 = solid_object.compute_inverse_dyn(q0, qd0, qdd0, T0)

        np.testing.assert_allclose(w1, np.r_[0, 0, 0, 0, 0, 9.8], atol=1e-8)
        assert np.allclose(w1, np.r_[0, 0, 0, 0, 0, 9.8])

def test_static_offset(setup):
    robot = setup
    link = robot.GetLink('link6')
    T_link_object = np.eye(4)
    dofindices = range(6)

    obj = toppra_app.SolidObject(robot, 'link6', T_link_object, 1, np.eye(3), dofindices=range(6))

    q0 = np.zeros(6)
    qd0 = np.zeros(6)
    qdd0 = np.zeros(6)

    with robot:
        robot.SetDOFValues(q0, dofindices)
        p_com = link.GetTransform()[:3, 3]

    T_test = np.eye(4)
    r = np.r_[0.2, 1, 0.5]
    T_test[:3, 3] = p_com + r

    w1 = obj.compute_inverse_dyn(q0, qd0, qdd0, T_test)

    np.testing.assert_allclose(w1[:3], - np.cross(r, np.r_[0, 0, 9.8]))
    np.testing.assert_allclose(w1[3:], np.r_[0, 0, 9.8])

def test_dynamics_coincide(setup):
    robot = setup
    link = robot.GetLink('link6')
    T_link_object = np.eye(4)
    dofindices = range(6)

    obj = toppra_app.SolidObject(robot, 'link6', T_link_object, 1, np.eye(3), dofindices=range(6))

    np.random.seed(10)
    q0 = np.random.randn(6) / 2
    qd0 = np.random.randn(6) / 2
    qdd0 = np.random.randn(6) / 2

    with robot:
        robot.SetDOFValues(q0, dofindices)
        T_link = link.GetTransform()
        J_rot, J_tran = compute_Jacobians(robot, 'link6', q0, T_link[:3, 3])
        H_rot, H_tran = compute_Hessians(robot, 'link6', q0, T_link[:3, 3])

        a = J_tran.dot(qdd0) + np.dot(qd0, np.dot(H_tran, qd0))
        alpha = J_rot.dot(qdd0) + np.dot(qd0, np.dot(H_rot, qd0))
        omega = J_rot.dot(qd0)

        I = T_link[:3, :3].dot(np.eye(3)).dot(T_link[:3, :3].T)

        f = a + np.r_[0, 0, 9.8]
        m = I.dot(alpha) + np.cross(omega, I.dot(omega))

        T_com = np.eye(4)
        T_com[:3, 3] = T_link[:3, 3]

    w1 = obj.compute_inverse_dyn(q0, qd0, qdd0, T_com)

    np.testing.assert_allclose(w1[3:], f, atol=1e-8)
    np.testing.assert_allclose(w1[:3], m, atol=1e-8)



def test_dynamics_noncoincide(setup):
    robot = setup
    link = robot.GetLink('link6')
    T_link_object = np.array([[1, 0, 0, 0.4],
                              [0, 0, -1, 0.2],
                              [0, 1, 0, -0.2],
                              [0, 0, 0, 1]])
    dofindices = range(6)
    obj = toppra_app.SolidObject(robot, 'link6', T_link_object, 1, np.eye(3), dofindices=range(6))

    np.random.seed(10)
    q0 = np.random.randn(6) / 2
    qd0 = np.random.randn(6) / 2
    qdd0 = np.random.randn(6) / 2

    with robot:
        robot.SetDOFValues(q0, dofindices)
        T_link = link.GetTransform()
        T_object = T_link.dot(T_link_object)
        J_rot, J_tran = compute_Jacobians(robot, 'link6', q0, T_object[:3, 3])
        H_rot, H_tran = compute_Hessians(robot, 'link6', q0, T_object[:3, 3])

        a = J_tran.dot(qdd0) + np.dot(qd0, np.dot(H_tran, qd0))
        alpha = J_rot.dot(qdd0) + np.dot(qd0, np.dot(H_rot, qd0))
        omega = J_rot.dot(qd0)

        I = T_object[:3, :3].dot(np.eye(3)).dot(T_object[:3, :3].T)

        f = a + np.r_[0, 0, 9.8]
        m = I.dot(alpha) + np.cross(omega, I.dot(omega))

        T_com = np.eye(4)
        T_com[:3, 3] = T_object[:3, 3]

    w1 = obj.compute_inverse_dyn(q0, qd0, qdd0, T_com)

    np.testing.assert_allclose(w1[3:], f, atol=1e-8)
    np.testing.assert_allclose(w1[:3], m, atol=1e-8)


def test_dynamics_noncoincide_body_frame(setup):
    robot = setup
    link = robot.GetLink('link6')
    T_link_object = np.array([[1, 0, 0, 0.4],
                           [0, 0, -1, 0.2],
                           [0, 1, 0, -0.2],
                           [0, 0, 0, 1]])
    dofindices = range(6)
    obj = toppra_app.SolidObject(robot, 'link6', T_link_object, 1, np.eye(3), dofindices=range(6))

    np.random.seed(2)
    q0 = np.random.randn(6) / 2
    qd0 = np.random.randn(6) / 2
    qdd0 = np.random.randn(6) / 2

    with robot:
        robot.SetDOFValues(q0, dofindices)
        T_link = link.GetTransform()
        T_object = T_link.dot(T_link_object)
        J_rot, J_tran = compute_Jacobians(robot, 'link6', q0, T_object[:3, 3])
        H_rot, H_tran = compute_Hessians(robot, 'link6', q0, T_object[:3, 3])

        a = J_tran.dot(qdd0) + np.dot(qd0, np.dot(H_tran, qd0))
        alpha = J_rot.dot(qdd0) + np.dot(qd0, np.dot(H_rot, qd0))
        omega = J_rot.dot(qd0)

        I = T_object[:3, :3].dot(np.eye(3)).dot(T_object[:3, :3].T)

        f = a + np.r_[0, 0, 9.8]
        m = I.dot(alpha) + np.cross(omega, I.dot(omega))

        R_world_object = T_object[:3, :3]
        f_body = R_world_object.T.dot(f)
        m_body = R_world_object.T.dot(m)

    T_body = obj.compute_frame_transform(q0)
    w1 = obj.compute_inverse_dyn(q0, qd0, qdd0, T_body)

    np.testing.assert_allclose(w1[3:], f_body, atol=1e-8)
    np.testing.assert_allclose(w1[:3], m_body, atol=1e-8)

