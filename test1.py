"""
This script works given a simple example

"""
import numpy as np
import openravepy as orpy
import time
import toppra as ta
import matplotlib.pyplot as plt

import quadprog

from toppra import SUCCESSFUL_RETURN, logger, SUPERTINY
import coloredlogs

coloredlogs.install(level='INFO')
np.set_printoptions(3)


def compute_Jacobians(robot, endlink_name, q_cur, pos_world):
    """

    Parameters
    ----------
    robot: openravepy.Robot
    endlink_name: str
        Name of the end-effector link, or the link that the object is attached to.
    q_cur: ndarray
        Joint pos_endlink to compute those Jacobians at.
    pos_endlink: ndarray
        Position of the object's com in the end-effector link.

    Returns
    -------
    J_rot: ndarray
    J_trans: ndarray
    """
    with robot:
        manip = robot.GetActiveManipulator()
        robot.SetDOFValues(q_cur, manip.GetArmIndices())
        link = robot.GetLink(endlink_name)
        J_trans = robot.ComputeJacobianTranslation(link.GetIndex(), pos_world, manip.GetArmIndices())
        J_rot = robot.ComputeJacobianAxisAngle(link.GetIndex(), manip.GetArmIndices())
    return J_rot, J_trans


def compute_Hessians(robot, endlink_name, q_cur, pos_world):
    """

    Parameters
    ----------
    robot: openravepy.Robot
    endlink_name: str
        Name of the end-effector link, or the link that the object is attached to.
    q_cur: ndarray
        Joint pos_endlink to compute those Jacobians at.
    pos_endlink: ndarray
        Position of the object's com in the end-effector link.

    Returns
    -------
    H_rot: ndarray
    H_trans: ndarray
    """
    with robot:
        manip = robot.GetActiveManipulator()
        arm_indices = manip.GetArmIndices()
        robot.SetDOFValues(q_cur, manip.GetArmIndices())
        link = robot.GetLink(endlink_name)
        J_trans = robot.ComputeHessianTranslation(link.GetIndex(), pos_world, arm_indices)
        J_rot = robot.ComputeHessianAxisAngle(link.GetIndex(), arm_indices)
    return J_rot, J_trans


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


class EpsilonPathParameterizationSolver(ta.qpOASESPPSolver):
    def solve_epsilon_pp(self, epsilon=1e-2, solver='qpOASES'):
        """ Compute Epsilon Path-Parameterization.

        Returns
        -------
        us: ndarray or None
        xs: ndarray or None
        """
        # Backward pass
        controllable = self.solve_controllable_sets()
        # Check controllability
        infeasible = (self._K[0, 1] < self.I0[0] or self._K[0, 0] > self.I0[1])

        if not controllable or infeasible:
            logger.warn("""
Unable to parameterizes this path:
- K(0) is empty : {0}
- sd_start not in K(0) : {1}
""".format(controllable, infeasible))
            return None, None

        # Forward pass
        xs = np.zeros(self.N + 1)
        us = np.zeros(self.N)
        xs[0] = (self.I0[0] + self.I0[1]) / 2
        for i in range(self.N):
            if i == 0:
                use_init = True
            else:
                use_init = False
            x_next_target = max(np.mean(self._K[i + 1]), self._K[i + 1, 1] - epsilon)
            g = np.array([4 * self.Ds[i] * (xs[i] - x_next_target), 0])
            H = np.eye(2)
            H[1, 1] = 1.0
            H[0, 0] = 4 * self.Ds[i] ** 2 + 1e-2
            # u_, x_ = self.forward_step(i, xs[i], self._K[i + 1, 0], self._K[i + 1, 1], H, g, init=use_init)
            u_, x_ = self.forward_step_quadprog(i, xs[i], self._K[i + 1, 0], self._K[i + 1, 1], H, g, init=use_init)
            if u_ is None:
                logger.warn("Forward pass fails. This should not happen.")
                return us, xs
            xs[i + 1] = x_
            us[i] = u_
        return us, xs

    def forward_step(self, i, x, x_next_min, x_next_max, H, g, init=False):
        """ Take a forward quadratic step from position s[i], state x.

        This step is taken by solving a quadratic program.  The objective function is given by

        .. math::
            \\text{min} \; \mathbf{g}^\\top [u, x]^\\top + 0.5 [u, x] \mathbf{H} [u, x]^\\top,
        Constraints include the problem constraints and the constraint

        .. math::
            x_{next, min} \leq 2 \Delta u + x \leq x_{next, max}
        For some reasons, `qpOASES` does not work very well; hence, this function uses `quadprog`
        to handle the Quadratic Programs instead.

        Parameters
        ----------
        i: int
        x: float
        x_next_min: float
        x_next_max: float
        H: ndarray
        g: ndarray
        init: bool, optional
            If is true, start the solver without using previously saved result. Else, start the solver using previously
            saved result.

        Returns
        -------
        u_next: float
            If infeasible, returns None.
        x_next: float
            If infeasible, returns None.

        """
        self.reset_operational_rows()
        quadprog.solve_qp

        # Enforce x == xs[i]
        self.A[i, 0, 1] = 1.
        self.A[i, 0, 0] = 0.
        self.lA[i, 0] = x
        self.hA[i, 0] = x
        # Constraint 2: x_next_min <= 2 ds u + x <= x_next_max
        self.lA[i, 1] = x_next_min
        self.hA[i, 1] = x_next_max
        self.A[i, 1, 1] = 1.
        self.A[i, 1, 0] = 2 * self.Ds[i]

        nWSR_topp = np.array([self.nWSR_cnst])  # The number of "constraint flipping"

        if init:
            # "Manual" warmup
            res_up = self.solver_up.init(
                np.eye(2) * 10.0, g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], np.array([1000]))
            res_up = self.solver_up.hotstart(
                H, g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_topp)
        else:
            res_up = self.solver_up.hotstart(
                H, g, self.A[i], self.l[i], self.h[i], self.lA[i],
                self.hA[i], nWSR_topp)

        if (res_up != SUCCESSFUL_RETURN):
            logger.warn("Non-optimal solution at i=%d. Returning default.", i)
            return None, None

        # extract solution
        self.solver_up.getPrimalSolution(self._xfull)
        # self.solver_up.getDualSolution(self._yfull)  # cause failure
        u_next = self._xfull[0]
        x_next = x + 2 * self.Ds[i] * u_next
        assert x_next + SUPERTINY >= 0, "Negative state (forward pass):={:f}".format(x_next)
        if x_next < 0:
            x_next = x_next + SUPERTINY
        return u_next, x_next

    def forward_step_quadprog(self, i, x, x_next_min, x_next_max, H, g, init=False):
        """ Take a forward quadratic step from position s[i], state x.

        This step is taken by solving a quadratic program.  The objective function is given by

        .. math::
            \\text{min} \; \mathbf{g}^\\top [u, x]^\\top + 0.5 [u, x] \mathbf{H} [u, x]^\\top,
        Constraints include the problem constraints and the constraint

        .. math::
            x_{next, min} \leq 2 \Delta u + x \leq x_{next, max}
        For some reasons, `qpOASES` does not work very well; hence, this function uses `quadprog`
        to handle the Quadratic Programs instead.

        Parameters
        ----------
        i: int
        x: float
        x_next_min: float
        x_next_max: float
        H: ndarray
        g: ndarray
        init: bool, optional
            If is true, start the solver without using previously saved result. Else, start the solver using previously
            saved result.

        Returns
        -------
        u_next: float
            If infeasible, returns None.
        x_next: float
            If infeasible, returns None.

        """
        self.reset_operational_rows()

        # Enforce x == xs[i]
        self.A[i, 0, 1] = 1.
        self.A[i, 0, 0] = 0.
        self.lA[i, 0] = x
        self.hA[i, 0] = x
        # Constraint 2: x_next_min <= 2 ds u + x <= x_next_max
        self.lA[i, 1] = x_next_min
        self.hA[i, 1] = x_next_max
        self.A[i, 1, 1] = 1.
        self.A[i, 1, 0] = 2 * self.Ds[i]

        # Parameters
        C_transpose = np.vstack((self.A[i], -self.A[i], np.eye(self.nV), -np.eye(self.nV)))
        b = np.hstack((self.lA[i], -self.hA[i], self.l[i], -self.h[i]))

        result = quadprog.solve_qp(H, -g, C_transpose.T, b)
        u_next = result[0][0]
        x_next = x + 2 * self.Ds[i] * u_next
        assert x_next + SUPERTINY >= 0, "Negative state (forward pass):={:f}".format(x_next)
        if x_next < 0:
            x_next = x_next + SUPERTINY
        return u_next, x_next


if __name__ == '__main__':
    env = orpy.Environment()
    env.Load('models/denso_ft_gripper_with_base.robot.xml')
    robot = env.GetRobots()[0]
    manip = robot.SetActiveManipulator('denso_ft_sensor')
    arm_indices = manip.GetArmIndices()
    ft_name = manip.GetEndEffector().GetName()
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
    # Contact properties
    file = np.load("data/alum_block_rubber_contact_contact.npz")
    A = file['A']
    b = file['b']

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
    pc_velocity = ta.create_velocity_path_constraint(path, ss, vlim)

    alim_ = robot.GetDOFAccelerationLimits(arm_indices) * acceleration_safety_factor
    alim = np.vstack((-alim_, alim_)).T
    pc_accel = ta.create_acceleration_path_constraint(path, ss, alim)

    q_vec = path.eval(ss)
    qs_vec = path.evald(ss)
    qss_vec = path.evaldd(ss)
    a_vec = []
    b_vec = []
    c_vec = []

    theta1_vec = []
    theta2_vec = []
    theta3_vec = []

    # At each position s, we now evaluate the coefficients of the
    # constraints.  See equation (xxx) in the paper. The final goal is
    # to evaluate three coefficients: theta1, theta2, theta3 before
    # concatenating the actual constraints.
    print "Form contact stability constraints"

    for i, s in enumerate(ss):
        robot.SetDOFValues(q_vec[i], arm_indices)
        T_we = manip.GetTransform()
        T_wo = T_we.dot(T_eo)
        R_wo = T_wo[:3, :3]
        p_wo = T_wo[:3, 3]

        R_wo_6 = np.zeros((6, 6))
        R_wo_6[:3, :3] = T_wo[:3, :3]
        R_wo_6[3:, 3:] = T_wo[:3, :3]

        I_w = R_wo.dot(I_o).dot(R_wo.T)
        J_rot, J_tran = compute_Jacobians(robot, ft_name, q_vec[i], p_wo)
        H_rot, H_tran = compute_Hessians(robot, ft_name, q_vec[i], p_wo)

        # kinematic parameters: translation
        # v = t1 * \dot s,  a = t1 * \ddot s + t2 * \dot s ^ 2
        t1_trans = J_tran.dot(qs_vec[i])
        t2_trans = J_tran.dot(qss_vec[i]) + np.dot(qs_vec[i], np.dot(H_tran, qs_vec[i]))

        # kinematic parameters: rotation
        # \omega = t1 * \dot s,  \alpha = t1 * \ddot s + t2 * \dot s ^ 2
        t1_rot = J_rot.dot(qs_vec[i])
        t2_rot = J_rot.dot(qss_vec[i]) + np.dot(qs_vec[i], np.dot(H_rot, qs_vec[i]))

        # thetas in world frame
        theta1_w = np.hstack((
            I_w.dot(t1_rot),
            m * t1_trans))

        theta2_w = np.hstack((
            I_w.dot(t2_rot) + np.cross(t1_rot, I_w.dot(t1_rot)),
            m * t2_trans
        ))

        theta3_w = - np.hstack((np.zeros(3), m * g_w))

        # thetas in object frame
        theta1 = R_wo_6.T.dot(theta1_w)
        theta2 = R_wo_6.T.dot(theta2_w)
        theta3 = R_wo_6.T.dot(theta3_w)

        # Finally, the set of feasible wrench : Aw <= b
        # After a simple manipulation: A(theta1 u + theta2 x + theta 3) <= b
        a_vec.append(A.dot(theta1))
        b_vec.append(A.dot(theta2))
        c_vec.append(A.dot(theta3) - b)

        theta1_vec.append(theta1)
        theta2_vec.append(theta2)
        theta3_vec.append(theta3)

    pc_wrench = ta.PathConstraint(a_vec, b_vec, c_vec, name="ContactWrench", ss=ss)

    np.savez("object_transport_test.npz",
             a_vec=a_vec, b_vec=b_vec, c_vec=c_vec,
             theta1=theta1_vec, theta2=theta2_vec, theta3=theta3_vec)

    print "Generate and solve TOPP!"
    pcs = [ta.interpolate_constraint(pc_velocity), ta.interpolate_constraint(pc_accel), pc_wrench]
    topp = EpsilonPathParameterizationSolver(pcs)
    topp.set_start_interval(0)
    topp.set_goal_interval(0)
    t0 = time.time()
    us, xs = topp.solve_epsilon_pp(epsilon=0.1)
    # us, xs = topp.solve_topp()

    t_elapsed = time.time() - t0
    print "Solve TOPP took: {:f} secs".format(t_elapsed)

    print "View the result!"
    plt.plot(topp.K[:, 0], '--', c='red')
    plt.plot(topp.K[:, 1], '--', c='red')
    plt.plot(xs, c='blue')
    plt.show()

    t_array, q_array, qdot_array, _ = ta.compute_trajectory_gridpoints(path, ss, us, xs)

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
