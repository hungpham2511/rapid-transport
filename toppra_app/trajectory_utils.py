import toppra
import cvxpy
import numpy as np

def generate_twist_at_active_conf(robot, max_angle=0.5):
    """ Generate a trajectory to realize a twisting motion about the y-axis
    of the active manipulator.

    Parameters
    ----------
    robot

    Returns
    -------
    traj_array: (N,dof)array

    """
    manip = robot.GetActiveManipulator()
    point = manip.GetTransform()[:3, 3]
    link = manip.GetEndEffector()
    qinit = robot.GetActiveDOFValues()
    J_trans = robot.ComputeJacobianTranslation(link.GetIndex(), point)
    J_rot = robot.ComputeJacobianAxisAngle(link.GetIndex())
    y_axis = manip.GetTransform()[:3, :3].dot([0, 1, 0])

    # Find dq such that J_trans dq ~ 0 and J_rot dq = y_axis
    dq_ = cvxpy.Variable(6)
    constraints = [qinit + max_angle * dq_ <= robot.GetDOFLimits()[1], qinit + max_angle * dq_ >= robot.GetDOFLimits()[0]]
    cost = 10 * cvxpy.norm(J_trans * dq_) ** 2 + cvxpy.norm(J_rot * dq_ - y_axis) ** 2 + cvxpy.norm(dq_) ** 2 * 0.1
    obj = cvxpy.Minimize(cost)
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()
    assert prob.status == "optimal", "Fail to solve the optimization problem."
    #

    dq_val = np.array(dq_.value).flatten()
    # Fill the blank
    traj_array = []
    for s in np.arange(0, -max_angle, -0.01):
        traj_array.append(qinit + dq_val * s)
    for s in np.arange(-max_angle, max_angle, 0.01):
        traj_array.append(qinit + dq_val * s)
    for s in np.arange(max_angle, -0.001, -0.01):
        traj_array.append(qinit + dq_val * s)

    return np.array(traj_array)
