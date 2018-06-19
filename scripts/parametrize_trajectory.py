import toppra, transport, hashlib, yaml
import toppra.algorithm
import numpy as np
import argparse, os, time
import openravepy as orpy
import matplotlib.pyplot as plt
import cvxpy as cvx

def main():
    parse = argparse.ArgumentParser(description="A program for parametrizing trajectory. Output trajectory"
                                    "id is generated from the ids of contact, object, robot and"
                                    "algorithm respectively using a hash function. Hence, if two"
                                    "trajectories are parametrized using the same setup, their ids"
                                    "will be similar."
                                    "")
    parse.add_argument('-c', '--contact', help='Profile id of the contact model', required=True)
    parse.add_argument('-o', '--object', help='Profile id of the object to transport', required=True)
    parse.add_argument('-T', '--transform', help='Transform from {link} to {object}: T_link_object', required=False, default="[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 9.080e-3], [0, 0, 0, 1]]")
    parse.add_argument('-r', '--robot', help='Robot specification. Contain file path to openrave robot model, also its velocity and acceleration limits.', required=False, default="suctioncup1")
    parse.add_argument('-l', '--algorithm', help='Algorithm specification.', default="topp_fast")
    parse.add_argument('-t', '--trajectory', help='Input trajectory specification.', required=False)
    parse.add_argument('-v', '--verbose', help='More verbose output', action="store_true")
    args = vars(parse.parse_args())

    if args['verbose']:
        import coloredlogs
        coloredlogs.install(level='DEBUG')
        np.set_printoptions(3)
    else:
        import coloredlogs
        coloredlogs.install(level='INFO')

    # Finish parsing, load different profiles
    db = transport.database.Database()
    contact_profile = db.retrieve_profile(args['contact'], "contact")
    object_profile = db.retrieve_profile(args['object'], "object")
    traj_profile = db.retrieve_profile(args['trajectory'], "trajectory")
    algorithm_profile = db.retrieve_profile(args['algorithm'], "algorithm")
    robot_profile = db.retrieve_profile(args['robot'], "robot")
    N = algorithm_profile['N']

    # Setup necessary modules and objects
    env = orpy.Environment()
    env.Load(transport.utils.expand_and_join(db.get_model_dir(), robot_profile['robot_model']))
    robot = env.GetRobots()[0]
    manip = robot.GetManipulator(robot_profile['manipulator'])
    arm_indices = manip.GetArmIndices()

    T_object = np.array(yaml.load(args['transform']), dtype=float)
    solid_object = transport.SolidObject(robot, object_profile['attached_to_manipulator'],
                                         T_object,
                                         object_profile['mass'],
                                         np.array(object_profile['local_inertia'], dtype=float),
                                         dofindices=arm_indices)
    contact = transport.Contact.init_from_profile_id(robot, args['contact'])
    assert contact.F_local is not None, "A contact needs to be pre-processed before it can be used. Run simplify_wrench.py on this contact."

    # Setup constraints: contact(object tranport), velocity and acceleration
    pc_object_trans = transport.create_object_transporation_constraint(contact, solid_object)
    if algorithm_profile['interpolate_dynamics']:
        pc_object_trans.set_discretization_type(1)
    else:
        pc_object_trans.set_discretization_type(0)

    print pc_object_trans
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

    # retrieve and define geometric path
    if "waypoints_npz" in traj_profile:
        file_ = np.load(os.path.join(db.get_trajectory_data_dir(), traj_profile['waypoints_npz']))
        ts_waypoints = file_['t_waypoints']
        waypoints = file_['waypoints']
    elif "t_waypoints" in traj_profile:
        ts_waypoints = traj_profile['t_waypoints']
        waypoints = traj_profile['waypoints']
    else:
        raise IOError, "Waypoints not found in trajectory {:}".format(traj_profile['id'])

    # setup toppra instance
    path = toppra.SplineInterpolator(ts_waypoints, waypoints)
    ss = np.linspace(0, path.get_duration(), N + 1)

    instance = toppra.algorithm.TOPPRA([pc_accel, pc_velocity, pc_object_trans], path, ss,
                                       solver_wrapper=algorithm_profile['solver_wrapper'])

    # solve first stage: toppra
    t0 = time.time()
    _, sd_vec, _ = instance.compute_parameterization(0, 0)
    xs = sd_vec ** 2
    print "solve TOPP with TOPP-RA takes {:f} seconds".format(time.time() - t0)

    # solve second stage: post-processing
    t0 = time.time()
    ds = 1.0 / N
    sddd_bnd = 200
    # matrix for computing jerk: 
    # Jmat = [-2  1 .....  ]
    #        [ 1 -2  1 ... ]
    #        [ 0  1 -2 1 ..]
    Jmat = np.zeros((N+1, N+1))   
    for i in range(N+1):
        Jmat[i, i] = -2
    for i in range(N):
        Jmat[i + 1, i] = 1
        Jmat[i, i + 1] = 1
    Jmat = Jmat / ds ** 2
     
    # matrix for differentiation
    Amat = np.zeros((N, N + 1))
    for i in range(N):
        Amat[i, i] = -1
        Amat[i, i + 1] = 1
    Amat = Amat / 2 / ds
   
    xs_var = cvx.Variable(N + 1)
    x_pprime_var = Jmat * xs_var
    s_dddot_var = cvx.mul_elemwise(np.sqrt(xs) + 1e-5, x_pprime_var)
    residue = cvx.Variable()
    
    constraints = [xs_var >= 0,
                   xs_var[0] == 0,
                   xs_var[-1] == 0,
                   xs_var <= xs,
                   0 <= residue,
                   s_dddot_var - sddd_bnd <= residue,
                   -sddd_bnd - s_dddot_var <= residue]

    obj = cvx.Minimize(1.0 / N * cvx.norm(xs_var - xs, 1)
                       + 1.0 / N * cvx.norm(Amat.dot(xs) - Amat * xs_var, 1)
                       + 0.0 * residue)
    prob = cvx.Problem(obj, constraints)
    status = prob.solve()

    print "post-processing takes {:f} seconds".format(time.time() - t0)
    print "Problem status {:}".format(status)
    print "residual value: {:f}".format(residue.value)
    xs_new = np.array(xs_var.value).flatten()
    xs_new[0] = 0
    xs_new[-1] = 0

    # Compute trajectory
    ss = np.linspace(0, 1, N + 1)
    ts = [0]
    for i in range(1, N + 1):
        sd_ = (np.sqrt(xs_new[i]) + np.sqrt(xs_new[i - 1])) / 2
        ts.append(ts[-1] + (ss[i] - ss[i - 1]) / sd_)
    q_grid = path.eval(ss)
    # joint_traj = toppra.SplineInterpolator(ts, q_grid, bc_type='natural')
    joint_traj = toppra.SplineInterpolator(ts, q_grid)

    try:
        print("Trajectory duration: {:f} seconds".format(joint_traj.get_duration()))
    except Exception as e:
        print("Unable to parametrize trajectory!")

    # Plot the velocity profile for debugging purpose
    if args['verbose']:
        ymax = 3.5
        K = instance.compute_controllable_sets(0, 0)
        plt.plot(K[:, 0], c='red', label="controllable sets")
        plt.plot(K[:, 1], c='red')
        try:
            plt.plot(xs, '--', c='green', label="velocity profile (w/out pp)")
            plt.plot(xs_new, c='blue', label="velocity profiel (with pp)")
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
        plt.legend()
        plt.show()

        try:
            # Joint trajectory
            t_samples = np.arange(0, joint_traj.get_duration(), algorithm_profile['sampling_period'])
            q_samples = joint_traj.eval(t_samples)
            qd_samples = joint_traj.evald(t_samples)
            qdd_samples = joint_traj.evaldd(t_samples)
            fig, axs = plt.subplots(3, 1, sharex=True)
            axs[0].plot(t_samples, q_samples)
            axs[0].set_title("Joint position (t)")
            axs[1].plot(t_samples, qd_samples)
            axs[1].set_title("Joint velocity (t)")
            axs[2].plot(t_samples, qdd_samples)
            axs[2].set_title("Joint acceleration (t)")
            plt.show()
        except Exception as e:
            pass

    identify_string = str(args['contact'] + args['object'] + args['algorithm'] + args['robot'] + args['transform'])
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
            'reparametrized': True,
            'reparam_trajectory': args['trajectory'],
            'reparam_object': args['object'],
            'reparam_transform': args['transform'],
            'reparam_contact': args['contact'],
            'reparam_robot': args['robot'],
            'waypoints_npz': traj_param_id+".npz"
        }
        db.insert_profile(traj_param_profile, "trajectory")
        print("Insert new profile!")
        exit()


if __name__ == '__main__':
    main()
