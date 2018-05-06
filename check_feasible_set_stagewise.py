import toppra, toppra_app
import toppra.algorithm
import numpy as np
import argparse, os, time
import openravepy as orpy
import matplotlib.pyplot as plt


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

    constraint = toppra.constraint.CanonicalLinearSecondOrderConstraint(inv_dyn, cnst_F, cnst_g)
    return constraint


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="A program for parametrizing trajectory. ")
    parse.add_argument('-c', '--contact', help='Id of the contact to be simplified', required=True)
    parse.add_argument('-o', '--object', help='Id of the object to transport', required=True)
    parse.add_argument('-r', '--robot', help='Robot specification.', required=True)
    parse.add_argument('-a', '--algorithm', help='Algorithm specification.', required=False)
    parse.add_argument('-t', '--trajectory', help='Input trajectory specification.', required=False)
    parse.add_argument('-v', '--verbose', help='More verbose output', action="store_true")
    args = vars(parse.parse_args())

    if args['verbose']:
        import coloredlogs
        coloredlogs.install(level='DEBUG')
        np.set_printoptions(3)

    db = toppra_app.database.Database()

    contact_profile = db.retrieve_profile(args['contact'], "contact")
    object_profile = db.retrieve_profile(args['object'], "object")
    traj_profile = db.retrieve_profile(args['trajectory'], "trajectory")
    algorithm_profile = db.retrieve_profile(args['algorithm'], "algorithm")
    robot_profile = db.retrieve_profile(args['robot'], "robot")

    # check that all input use the same robot model
    assert contact_profile['attached_to_robot'] == robot_profile['robot_model']
    assert object_profile['attached_to_robot'] == robot_profile['robot_model']
    assert traj_profile['attached_to_robot'] == robot_profile['robot_model']

    env = orpy.Environment()
    env.Load('models/' + robot_profile['robot_model'])
    robot = env.GetRobots()[0]
    manip = robot.GetManipulator(robot_profile['manipulator'])
    arm_indices = manip.GetArmIndices()

    T_object = np.eye(4)
    T_object[:3, 3] = object_profile['position']
    T_object[:3, :3] = object_profile['orientation']
    solid_object = toppra_app.SolidObject(robot, object_profile['attached_to_manipulator'],
                                          T_object,
                                          object_profile['mass'],
                                          object_profile['local_inertia'],
                                          dofindices=arm_indices)

    contact_data_file = np.load(os.path.join(db.get_contact_data_dir(), contact_profile['constraint_coeffs_file']))
    A = contact_data_file['A']  # constraints have the form A x <= b
    b = contact_data_file['b']
    T_contact = np.eye(4)
    T_contact[:3, 3] = contact_profile['position']
    T_contact[:3, :3] = contact_profile['orientation']
    contact = toppra_app.Contact(robot, contact_profile['attached_to_manipulator'],
                                 T_contact, A, b, dofindices=arm_indices)

    pc_object_trans = create_object_transporation_constraint(contact, solid_object)
    if algorithm_profile['interpolate_dynamics']:
        pc_object_trans.set_discretization_type(1)
    else:
        pc_object_trans.set_discretization_type(0)

    print "Assembling the constraints"
    vlim_ = np.r_[robot_profile['velocity_limits']]
    alim_ = np.r_[robot_profile['acceleration_limits']]
    vlim = np.vstack((-vlim_, vlim_)).T
    alim = np.vstack((-alim_, alim_)).T

    pc_velocity = toppra.constraint.JointVelocityConstraint(vlim)
    pc_accel = toppra.constraint.JointAccelerationConstraint(alim)
    if algorithm_profile['interpolate_kinematics']:
        pc_velocity.set_discretization_type(1)
        pc_accel.set_discretization_type(1)
    else:
        pc_velocity.set_discretization_type(0)
        pc_accel.set_discretization_type(0)

    path = toppra.SplineInterpolator(traj_profile['t_waypoints'], traj_profile['waypoints'])
    ss = np.linspace(0, path.get_duration(), algorithm_profile['N'] + 1)

    # instance = toppra.algorithm.TOPPRA([pc_accel, pc_velocity, pc_object_trans], path, ss, solver_wrapper=algorithm_profile['solver_wrapper'])
    instance = toppra.algorithm.TOPPRA([pc_object_trans], path, ss, solver_wrapper=algorithm_profile['solver_wrapper'])

    solver_wrapper = instance.solver_wrapper

    index = 40

    pts = []
    for i in range(20):
        v = toppra_app.utils.generate_random_unit_vectors(2, 1)[0]
        pt = solver_wrapper.solve_stagewise_optim(index, None, v, None, None, None, None)
        pts.append(pt)
    hull = toppra_app.poly_contact.ConvexHull(pts)

    vertices = np.vstack((hull.vertices, hull.vertices[0]))
    plt.plot(vertices[:, 0], vertices[:, 1])
    plt.title(contact_profile['id'])
    plt.grid()
    plt.show()



