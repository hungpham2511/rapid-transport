"""
This script is similar to test1.py, but utilizes the new toppra api.

"""
import numpy as np
import openravepy as orpy
import time
import toppra as ta
import matplotlib.pyplot as plt

from toppra_app.utils import compute_Hessians, compute_Jacobians
import toppra.constraint as constraint
import toppra.algorithm as algorithm
import toppra_app
import quadprog

import coloredlogs
coloredlogs.install(level='DEBUG')
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


def visualize_result(t_array, q_array, qdot_array):
    fig, axs = plt.subplots(2, 1, sharex=True)
    for i in range(6):
        axs[0].plot(t_array, q_array[:, i], label='J{:d}'.format(i+1))
        axs[1].plot(t_array, qdot_array[:, i], label='J{:d}'.format(i+1))
    axs[0].legend()
    axs[0].set_title('Joint Position')
    axs[1].legend()
    axs[1].set_title('Joint Velocity')
    axs[0].set_xlabel('Time')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    env = orpy.Environment()
    env.Load('models/denso_ft_gripper_with_base.robot.xml')
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
    file = np.load("data/alum_block_rubber_contact_contact.npz")
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

    print "Assembling the constraints"
    vlim_ = robot.GetDOFVelocityLimits(arm_indices) * velocity_safety_factor
    vlim = np.vstack((-vlim_, vlim_)).T
    pc_velocity = constraint.JointVelocityConstraint(vlim)

    alim_ = robot.GetDOFAccelerationLimits(arm_indices) * acceleration_safety_factor
    alim = np.vstack((-alim_, alim_)).T
    pc_accel = constraint.JointAccelerationConstraint(alim)

    # instance = algorithm.TOPPRA([pc_accel, pc_velocity, pc_object_trans], path, ss)
    instance = algorithm.TOPPRA([pc_accel, pc_velocity], path)

    print("Start solving constraints.")
    t0 = time.time()
    _, sd_vec, _ = instance.compute_parameterization(0, 0)
    xs = sd_vec ** 2
    t_elapsed = time.time() - t0
    print "Solve TOPP took: {:f} secs".format(t_elapsed)

    K = instance.compute_controllable_sets(0, 0)

    print "View the result!"
    plt.plot(K[:, 0], '--', c='red')
    plt.plot(K[:, 1], '--', c='red')
    plt.plot(xs, c='blue')
    plt.show()

    traj, _ = instance.compute_trajectory(0, 0)
    t_array = np.arange(0, traj.get_duration(), 1.0 / 150)
    q_array = traj.eval(t_array)
    qdot_array = traj.evald(t_array)

    visualize_result(t_array, q_array, qdot_array)

    # Preview trajectory
    dt = 1.0 / 150
    spline = ta.SplineInterpolator(t_array, q_array)
    t_uniform = np.arange(t_array[0], t_array[-1], dt)
    q_uniform = spline.eval(t_uniform)
    print "Start trajectory preview"
    for q in q_uniform:
        robot.SetDOFValues(q, manip.GetArmIndices())
        time.sleep(dt)

    # Save trajectory
    trajectory_name = 'test_trajectory_1'
    np.savez('trajectories/{:}.npz'.format(trajectory_name), q_uniform=q_uniform)

    import IPython
    if IPython.get_ipython() is None:
        IPython.embed()
