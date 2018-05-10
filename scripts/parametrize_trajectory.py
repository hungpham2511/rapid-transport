import toppra, toppra_app, hashlib, yaml
import toppra.algorithm
import numpy as np
import argparse, os, time
import openravepy as orpy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="A program for parametrizing trajectory. Output trajectory"
                                                "id is generated from the ids of contact, object, robot and"
                                                "algorithm respectively using a hash function. Hence, if two"
                                                "trajectories are parametrized using the same setup, their ids"
                                                "will be similar.")
    parse.add_argument('-c', '--contact', help='Id of the contact to be simplified', required=True)
    parse.add_argument('-o', '--object', help='Id of the object to transport', required=True)
    parse.add_argument('-a', '--attach', help='Name of the link or mnaipulator that the object is attached to.', required=False, default="denso_suction_cup")
    parse.add_argument('-T', '--transform', help='T_link_object', required=False, default="[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 9.080e-3], [0, 0, 0, 1]]")
    parse.add_argument('-r', '--robot', help='Robot specification.', required=False, default="suctioncup1")
    parse.add_argument('-l', '--algorithm', help='Algorithm specification.', default="topp_fast")
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
    assert contact_profile['attached_to_robot'] == robot_profile['robot_model'], "Supplied contact and robot must share the same robot model!"
    assert object_profile['attached_to_robot'] == robot_profile['robot_model'], "Supplied object and robot must share the same robot model!"
    assert traj_profile['attached_to_robot'] == robot_profile['robot_model'], "Supplied trajectory and robot must share the same robot model!"

    env = orpy.Environment()
    env.Load(toppra_app.utils.expand_and_join(db.get_model_dir(), robot_profile['robot_model']))
    robot = env.GetRobots()[0]
    manip = robot.GetManipulator(robot_profile['manipulator'])
    arm_indices = manip.GetArmIndices()

    T_object = np.array(yaml.load(args['transform']), dtype=float)
    solid_object = toppra_app.SolidObject(robot, object_profile['attached_to_manipulator'],
                                          T_object,
                                          object_profile['mass'],
                                          np.array(object_profile['local_inertia'], dtype=float),
                                          dofindices=arm_indices)
    contact = toppra_app.Contact.init_from_profile_id(robot, args['contact'])

    pc_object_trans = toppra_app.create_object_transporation_constraint(contact, solid_object)
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

    instance = toppra.algorithm.TOPPRA([pc_accel, pc_velocity, pc_object_trans], path, ss,
                                       solver_wrapper=algorithm_profile['solver_wrapper'])
    t0 = time.time()
    joint_traj, aux_traj, out = instance.compute_trajectory(0, 0, return_profile=True)
    print "Trajectory computation takes {:f} seconds".format(time.time() - t0)
    try:
        print("Trajectory duration: {:f} seconds".format(joint_traj.get_duration()))
    except Exception as e:
        print("Unable to parametrize trajectory!")

    # Plot the velocity profile for debugging purpose
    if args['verbose']:
        ymax = 3.5
        K = instance.compute_controllable_sets(0, 0)
        plt.plot(K[:, 0], c='red')
        plt.plot(K[:, 1], c='red')
        try:
            sdd_grid, sd_grid, _ = out
            xs = sd_grid ** 2
            plt.plot(xs, c='blue')
        except Exception as e:
            print("Parametrization fails. Do not plot velocity profile.")
        plt.xlabel("i: Discrete step")
        plt.ylabel("x=sd^2: Squared path velocity")
        plt.title("Velocity profile and controllable sets")
        plt.text(0, ymax + 0.1, "Trajectory: {:}\nContact: {:}".format(args['trajectory'], args['contact']),
                 horizontalalignment="left", verticalalignment="bottom")
        ymin, _ = plt.ylim()
        plt.ylim(ymin, ymax + 0.5)
        plt.grid()
        plt.show()

        try:
            # Joint trajectory
            t_samples = np.arange(0, joint_traj.get_duration(), algorithm_profile['sampling_period'])
            q_samples = joint_traj.eval(t_samples)
            plt.plot(t_samples, q_samples)
            plt.title("Joint position (t)")
            plt.show()
        except Exception as e:
            pass

    identify_string = str(
        args['contact'] + args['object'] + args['algorithm'] + args['robot'] + args['attach'] + args['transform']
    )
    traj_param_id = args['trajectory'] + "_" + hashlib.md5(identify_string).hexdigest()[:10]
    cmd = raw_input("Save parametrized trajectory to database as {:} y/[N]?".format(traj_param_id))
    if cmd != 'y':
        print("Exit without saving!")
        exit()
    else:
        np.savez(os.path.join(db.get_trajectory_data_dir(), traj_param_id+".npz"),
                 t_waypoints=joint_traj.ss_waypoints,
                 waypoints=joint_traj.waypoints)

        traj_param_profile = {
            'id': traj_param_id,
            'attached_to_robot': traj_profile['attached_to_robot'],
            'reparametrized': True,
            'reparam_trajectory': args['trajectory'],
            'reparam_object': args['object'],
            'reparam_attach': args['attach'],
            'reparam_transform': args['transform'],
            'reparam_contact': args['contact'],
            'reparam_robot': args['robot'],
            'waypoints_npz': traj_param_id+".npz"
        }
        db.insert_profile(traj_param_profile, "trajectory")
        print("Insert new profile!")
        exit()


