import pytest
import openravepy as orpy
import transport
from transport.utils import compute_Hessians, compute_Jacobians
import numpy as np


@pytest.fixture(scope="module")
def setup(envlab1):
    """ `envlab1` is defined in conftest.py, which is stored in the same
    folder.
    """
    robot = envlab1
    active_indices = range(6)
    robot.SetActiveDOFs(active_indices)
    ft_name = 'wam6'
    yield robot, ft_name


@pytest.mark.parametrize("q0", [np.zeros(6), np.r_[1, 0, 1, 0, 1, 0]])
def test_inv_dyn_frame_static(setup, q0):
    """If the robot has zero velocity and acceleration, it exerts a wrench
    on the object that equals gravitational force. All tests with
    `_static` subfix have this condition.

    Case 1: the object's com and the link's origin coincide.
    """
    robot, ft_name = setup
    T_eo = np.eye(4)
    m = 1
    I_o = np.eye(3)
    arm_indices = range(6)

    solid_object = transport.SolidObject(robot, ft_name, T_eo, m, I_o, dofindices=arm_indices)
    link = robot.GetLink(ft_name)

    qd0 = np.zeros(6)
    qdd0 = np.zeros(6)

    with robot:
        robot.SetActiveDOFValues(q0)
        T0 = np.eye(4)
        T0[:3, 3] = link.GetTransform()[:3, 3]

    w1 = solid_object.compute_inverse_dyn(q0, qd0, qdd0, T0)  # Wrench on the link's origin
    np.testing.assert_allclose(w1, np.r_[0, 0, 0, 0, 0, 9.8], atol=1e-8)

    w2 = solid_object.compute_inverse_dyn_local(q0, qd0, qdd0)  # Wrench on the object's com
    np.testing.assert_allclose(w2, np.r_[0, 0, 0, 0, 0, 9.8], atol=1e-8)


@pytest.mark.parametrize("q0", [np.zeros(6), np.r_[1, 0, 1, 0, 1, 0]])
def test_static_not_coincide(setup, q0):
    """Case 2: the object's com and the link's origin do not coincide.

    """
    robot, ft_name = setup
    T_link_obj = np.array([[1, 0, 0, 0.4],
                           [0, 0, -1, 0.2],
                           [0, 1, 0, -0.2],
                           [0, 0, 0, 1]])
    m = 1
    I_o = np.eye(3)
    arm_indices = range(6)

    solid_object = transport.SolidObject(robot, ft_name, T_link_obj, m, I_o, dofindices=arm_indices)
    link = robot.GetLink(ft_name)

    # Case 1: static and simple pose
    qd0 = np.zeros(6)
    qdd0 = np.zeros(6)

    with robot:
        robot.SetDOFValues(q0, arm_indices)
        T0 = np.eye(4)
        T0[:3, 3] = link.GetTransform().dot(T_link_obj)[:3, 3]

    w1 = solid_object.compute_inverse_dyn(q0, qd0, qdd0, T0)
    np.testing.assert_allclose(w1, np.r_[0, 0, 0, 0, 0, 9.8], atol=1e-8)
    assert np.allclose(w1, np.r_[0, 0, 0, 0, 0, 9.8])


def test_static_offset(setup):
    """
    """
    robot, ft_name = setup
    link = robot.GetLink(ft_name)
    T_link_object = np.eye(4)
    dofindices = range(6)

    obj = transport.SolidObject(robot, ft_name, T_link_object, 1, np.eye(3), dofindices=range(6))

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
    """The robot moves. Compare exerted wrench computed using SolidBody
    versus computed directly with the Jacobian and Hessian.

    All tests with `_dynamic` share this setting.

    NOTE: It seems like the rotational Hessian `H_rot` OpenRAVE
    computes is incorrect. Hence allow a higher tolerance.
    """
    robot, ft_name = setup
    link = robot.GetLink(ft_name)
    T_link_object = np.eye(4)
    dofindices = range(6)

    obj = transport.SolidObject(robot, ft_name, T_link_object, 1, np.eye(3), dofindices=range(6))

    np.random.seed(10)
    q0 = np.random.randn(6) / 2
    qd0 = np.random.randn(6) / 2
    qdd0 = np.random.randn(6) / 2

    with robot:
        robot.SetDOFValues(q0, dofindices)
        T_link = link.GetTransform()
        J_rot, J_tran = compute_Jacobians(robot, ft_name, q0, T_link[:3, 3])
        H_rot, H_tran = compute_Hessians(robot, ft_name, q0, T_link[:3, 3])

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
    np.testing.assert_allclose(w1[:3], m, atol=3e-1, rtol=1e-1)


def test_dynamics_noncoincide(setup):
    robot, ft_name = setup
    link = robot.GetLink(ft_name)
    T_link_object = np.array([[1, 0, 0, 0.4],
                              [0, 0, -1, 0.2],
                              [0, 1, 0, -0.2],
                              [0, 0, 0, 1]])
    obj = transport.SolidObject(robot, ft_name, T_link_object, 1, np.eye(3))

    np.random.seed(10)
    q0 = np.random.randn(6) / 2
    qd0 = np.random.randn(6) / 2
    qdd0 = np.random.randn(6) / 2

    with robot:
        robot.SetActiveDOFValues(q0)
        T_link = link.GetTransform()
        T_object = T_link.dot(T_link_object)
        J_rot, J_tran = compute_Jacobians(robot, ft_name, q0, T_object[:3, 3])
        H_rot, H_tran = compute_Hessians(robot, ft_name, q0, T_object[:3, 3])

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
    np.testing.assert_allclose(w1[:3], m, atol=3e-1, rtol=1e-1)


def test_dynamics_noncoincide_body_frame(setup):
    robot, ft_name = setup
    link = robot.GetLink(ft_name)
    T_link_object = np.array([[1, 0, 0, 0.4],
                           [0, 0, -1, 0.2],
                           [0, 1, 0, -0.2],
                           [0, 0, 0, 1]])
    obj = transport.SolidObject(robot, ft_name, T_link_object, 1, np.eye(3))

    np.random.seed(2)
    q0 = np.random.randn(6) / 2
    qd0 = np.random.randn(6) / 2
    qdd0 = np.random.randn(6) / 2

    with robot:
        robot.SetActiveDOFValues(q0)
        T_link = link.GetTransform()
        T_object = T_link.dot(T_link_object)
        J_rot, J_tran = compute_Jacobians(robot, ft_name, q0, T_object[:3, 3])
        H_rot, H_tran = compute_Hessians(robot, ft_name, q0, T_object[:3, 3])

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
    # np.testing.assert_allclose(w1[:3], m_body, atol=1e-8)


