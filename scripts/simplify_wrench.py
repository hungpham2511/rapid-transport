"""
The main goal of this script is to generate a set of inequalities
Ax <= b
where x is the set of feasible wrenches transformed to the Alum block's
body-fix frame.
"""
import numpy as np
import matplotlib.pyplot as plt
import toppra_app, hashlib
import argparse, yaml, os
import openravepy as orpy
from datetime import datetime

def preview_plot(args):
    """ Preview data tuples given in args.
    """
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title('tau_x vs tau_z')
    axs[0, 1].set_title('tau_y vs tau_z')
    axs[1, 0].set_title('f_x vs f_z')
    axs[1, 1].set_title('f_y vs f_z')
    for ws_all, marker, size in args:
        axs[0, 0].scatter(ws_all[:, 0], ws_all[:, 2], marker=marker, s=size)
        axs[0, 1].scatter(ws_all[:, 1], ws_all[:, 2], marker=marker, s=size)

        axs[1, 0].scatter(ws_all[:, 3], ws_all[:, 5], marker=marker, s=size)
        axs[1, 1].scatter(ws_all[:, 4], ws_all[:, 5], marker=marker, s=size)
    plt.show()


def strategy1(points):
    """ Simplify the data set, return the resulting data points.
    """
    hull_simplified = None
    max_hull_volume = -1
    for i in range(100):
        vca_indices = toppra_app.poly_contact.vertex_component_analysis(points)
        poly_indices = toppra_app.poly_contact.poly_expansion(points, vca_indices, 50, max_area=False)
        if poly_indices is None:
            continue
        hull3 = toppra_app.poly_contact.ConvexHull(points[poly_indices])
        vol = hull3.compute_volume()
        # print i, vol
        if vol > max_hull_volume:
            hull_simplified = hull3
            max_hull_volume = vol
    # print(hull_simplified.report())
    return hull_simplified.vertices


def strategy2(points):
    """ Simplify the data set, return the resulting data points.
    """
    hull_simplified = None
    max_hull_volume = -1
    for i in range(100):
        vca_indices = toppra_app.poly_contact.vertex_component_analysis(points)
        poly_indices = toppra_app.poly_contact.poly_expansion(points, vca_indices, 50, max_area=True)
        if poly_indices is None:
            continue
        hull3 = toppra_app.poly_contact.ConvexHull(points[poly_indices])
        vol = hull3.compute_volume()
        # print i, vol
        if vol > max_hull_volume:
            hull_simplified = hull3
            max_hull_volume = vol
    return hull_simplified.vertices


def strategy6(points, kp_indices):
    """ Simplify data set with polytopic expansion, using key points as initial seed.
    """
    hull_simplified = None
    max_hull_volume = -1
    for i in range(100):
        poly_indices = toppra_app.poly_contact.poly_expansion(points, kp_indices, 50, max_area=False)
        if poly_indices is None:
            continue
        hull3 = toppra_app.poly_contact.ConvexHull(points[poly_indices])
        vol = hull3.compute_volume()
        # print i, vol
        if vol > max_hull_volume:
            hull_simplified = hull3
            max_hull_volume = vol
    return hull_simplified.vertices


def strategy9(ws_all, N_vertices_max=100, N_samples=1000):
    "Polyhedral expansion with guidance from sampling"
    hull_full = toppra_app.poly_contact.ConvexHull(ws_all)
    print(hull_full.report())

    # %% thinning, choose 1000 samples
    print ("Start sampling")
    ws_thin = []
    for i in range(N_samples):
        ws_thin.append(toppra_app.poly_contact.uniform_interior_sampling(hull_full))
    ws_thin = np.array(ws_thin)
    print ("Finish sampling")

    # %% Polyhedral expansion
    vca_indices = toppra_app.poly_contact.vertex_component_analysis(ws_all)

    vertices = ws_all[vca_indices].tolist()
    vertices_index = list(vca_indices)
    N_vertices = -1
    while N_vertices < N_vertices_max:
        hull = toppra_app.poly_contact.ConvexHull(vertices)
        N_vertices = hull.vertices.shape[0]

        A, b = hull.get_halfspaces()
        # Select face to expand
        face_to_expand = None
        val = 1e-9
        for i in range(A.shape[0]):
            residues = ws_thin.dot(A[i]) - b[i]
            face_val = np.sum(residues[np.where(residues > 0)])
            if face_val > val:
                face_to_expand = i
                val = face_val

        if face_to_expand is None:
            # This mean all red vertices have been found
            print "Cover inner set!"
            break
        else:
            opt_vertex_index = np.argmax(ws_all.dot(A[face_to_expand]))

        vertices.append(ws_all[opt_vertex_index])
        vertices_index.append(opt_vertex_index)
    return vertices


def strategy10(ws_all, contact_profile, object_profile, robot_profile,
               N_vertices_max=50, Ad_T_wrench_transform=np.eye(6), N_samples=100):
    """ Simplify data points using DYNAMICALLY GUIDED EXPANSION.

    Parameters
    ----------
    ws_all: (N,6)array
    contact_profile: dict
    object_profile: dict
    robot_profile: dict
    N_vertices_max: int
        Number of maximum vertices.
    Ad_T_wrench_transform: (6,6)array
        A transformation to apply to each sampled wrench.

    Returns
    -------
    vertices: (M,6)array
    """
    hull_full = toppra_app.poly_contact.ConvexHull(ws_all)
    A, b = hull_full.get_halfspaces()
    print(hull_full.report())
    db = toppra_app.database.Database()

    env = orpy.Environment()
    env.Load(toppra_app.utils.expand_and_join(db.get_model_dir(), robot_profile['robot_model']))
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

    T_contact = np.eye(4)
    T_contact[:3, 3] = contact_profile['position']
    T_contact[:3, :3] = contact_profile['orientation']
    contact = toppra_app.Contact(robot, contact_profile['attached_to_manipulator'],
                                 T_contact, A, b, dofindices=arm_indices)

    # Sample points
    print ("Start sampling")
    ws_thin = []
    trial = 0
    while len(ws_thin) < N_samples:
        trial += 1
        qdd_sam, qd_sam = toppra_app.utils.sample_uniform(2, 0.5, 6)
        q_sam = toppra_app.utils.sample_uniform(1, 3, 6)[0]
        T_world_contact = contact.compute_frame_transform(q_sam)
        w_sam = solid_object.compute_inverse_dyn(q_sam, qd_sam, qdd_sam, T_world_contact)
        w_sam = Ad_T_wrench_transform.dot(w_sam)
        if np.all(A.dot(w_sam) - b <= 0):
            ws_thin.append(w_sam)
    ws_thin = np.array(ws_thin)
    print ("Finish sampling ({:d} trials / {:d} samples)".format(trial, N_samples))

    # %% Polyhedral expansion
    vca_indices = toppra_app.poly_contact.vertex_component_analysis(ws_all)

    vertices = ws_all[vca_indices].tolist()
    vertices_index = list(vca_indices)
    N_vertices = -1
    while N_vertices < N_vertices_max:
        hull = toppra_app.poly_contact.ConvexHull(vertices)
        N_vertices = hull.vertices.shape[0]

        A, b = hull.get_halfspaces()
        # Select face to expand
        face_to_expand = None
        val = 1e-9
        for i in range(A.shape[0]):
            residues = ws_thin.dot(A[i]) - b[i]
            face_val = np.sum(residues[np.where(residues > 0)])
            if face_val > val:
                face_to_expand = i
                val = face_val

        if face_to_expand is None:
            # This mean all red vertices have been found
            print "Cover inner set!"
            break
        else:
            opt_vertex_index = np.argmax(ws_all.dot(A[face_to_expand]))

        vertices.append(ws_all[opt_vertex_index])
        vertices_index.append(opt_vertex_index)
    vertices = np.array(vertices)
    fig, axs = plt.subplots(2, 2)

    to_plot = (
        (0, 1, axs[0, 0]),
        (0, 2, axs[0, 1]),
        (3, 4, axs[1, 0]),
        (4, 5, axs[1, 1]))

    for i, j, ax in to_plot:
        ax.scatter(ws_all[:, i], ws_all[:, j], c='C0', alpha=0.5, s=10)
        ax.scatter(ws_thin[:, i], ws_thin[:, j], marker='x', c='C1', zorder=10, s=50)
        ax.plot(vertices[:, i], vertices[:, j], c='C2')
    plt.show()

    return vertices


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="A program for simplifying and converting contact configurations. "
                                                "Contact should contain raw_data field.")
    parse.add_argument('-c', '--contact', help='Contact to be simplified', required=True)
    parse.add_argument('-s', '--strategy', help='Strategy to use. Best performing strategy so far: strategy10.', required=False, default="strategy10")
    parse.add_argument('-f', '--new_manip', help='New manipulator to transform contact to. Not fully support.', required=False)
    parse.add_argument('-o', '--object', help='Object specification, use for dynamic exploration (strategy10).', required=False)
    parse.add_argument('-r', '--robot', help='Robot specification, use for dynamic exploration (strategy10).', required=False, default="suctioncup1")
    parse.add_argument('-a', '--N_samples', help='Number of random wrench samples (strategy10)', required=False, default=100, type=int)
    parse.add_argument('-e', '--N_vertices', help='Number of max vertices during expansion (strategy10)', required=False, default=50, type=int)
    parse.add_argument('-v', '--verbose', help='More verbose output', action="store_true")
    args = vars(parse.parse_args())

    if args['verbose']:
        import coloredlogs
        coloredlogs.install(level='DEBUG')
        np.set_printoptions(3)

    db = toppra_app.database.Database()
    contact_profile = db.retrieve_profile(args['contact'], "contact")

    assert contact_profile['position'] == [0, 0, 0]
    assert contact_profile['orientation'] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    print("""
    Loaded contact:
    {:}
    """.format(contact_profile))
    if contact_profile['raw_data'] is None:
        raise ValueError, "Contact {:} does not have a 'raw_data' field, it is generated from some other contact. Quit."

    # Load data from text files
    try:
        ws_list = [contact_profile['key_points']]
        kp_indices = range(len(contact_profile['key_points']))
    except KeyError:
        ws_list = []
        kp_indices = []
    for file_name in contact_profile['raw_data']:
        file_dir = os.path.join(db.get_contact_data_dir(), file_name)
        ws_ = toppra_app.utils.load_data_ati_log(file_dir)
        ws_list.append(ws_)
    ws_all = np.vstack(ws_list)

    # Transform to new frame if specify
    if args['new_manip'] is not None:
        print "Loading robot!"
        env = orpy.Environment()
        env.Load("models/" + contact_profile['attached_to_robot'])
        robot = env.GetRobots()[0]
        manip = robot.GetManipulator(contact_profile['attached_to_manipulator'])
        T_world_old = manip.GetTransform()

        manip2 = robot.GetManipulator(args['new_manip'])
        if manip2 is None:
            raise ValueError, "Unable to find target frame {:} in the contact's robot model".format(args['new_manip'])
        T_world_new = manip2.GetTransform()

        T_old_new = toppra_app.utils.transform_inv(T_world_old).dot(T_world_new)
        Ad_T_old_new = toppra_app.utils.Ad(T_old_new)

        # wrench transform equation: w_a = Ad_T_ba^T w_b
        ws_all = ws_all.dot(Ad_T_old_new)
        new_manip = args['new_manip']
    else:
        new_manip = contact_profile['attached_to_manipulator']
        Ad_T_old_new = np.eye(6)  # Wrench transform

    # Preview scanned data
    preview_plot([[ws_all, 'x', 0.2], [ws_all[kp_indices], 'o', 10]])

    if args['strategy'] == "strategy1":
        points_simplified = strategy1(ws_all)
    elif args['strategy'] == "strategy2":
        points_simplified = strategy2(ws_all)
    elif args['strategy'] == "strategy5":
        points_simplified = strategy1(ws_all)
        points_simplified = np.vstack([points_simplified, ws_all[kp_indices]])
    elif args['strategy'] == "strategy6":
        points_simplified = strategy6(ws_all, kp_indices)
    elif args['strategy'] == "strategy9":
        points_simplified = strategy9(ws_all)
    elif args['strategy'] == "strategy9a":
        points_simplified = strategy9(ws_all, N_vertices_max=150)
        points_simplified = np.vstack([points_simplified, ws_all[kp_indices]])
    elif args['strategy'] == "strategy10":
        assert new_manip == contact_profile['attached_to_manipulator'], "Changing manipulator frame is not supported!"
        assert args['object'] is not None, "Object specification required!"
        assert args['robot'] is not None, "Robot specification required!"
        object_profile = db.retrieve_profile(args['object'], "object")
        robot_profile = db.retrieve_profile(args['robot'], "robot")
        points_simplified = strategy10(ws_all, contact_profile, object_profile, robot_profile,
                                       N_vertices_max=args['N_vertices'],
                                       Ad_T_wrench_transform=Ad_T_old_new,
                                       N_samples=args['N_samples'])
    else:
        raise NotImplementedError, "Strategy not found"

    hull = toppra_app.poly_contact.ConvexHull(ws_all)
    print(hull.report())

    hull_simplified = toppra_app.poly_contact.ConvexHull(points_simplified)
    print(hull_simplified.report())
    ws_new = hull_simplified.get_vertices()
    # Preview simplified wrenches
    preview_plot(([ws_all, 'x', 0.2], [ws_new, 'x', 10]))

    if args['strategy'] == "strategy10":
        new_contact_id = contact_profile['id'] + "_" \
                         + hashlib.md5(
            args['strategy'] + args['object'] + str(args['N_samples'])
            + str(args['N_vertices'])).hexdigest()[:10]
    else:
        new_contact_id = contact_profile['id'] + "_" \
                         + hashlib.md5(args['strategy']).hexdigest()[:10]
    cmd = raw_input("Save the simplified contact as [{:}] [y/N]?".format(new_contact_id))
    if cmd != 'y':
        print("Do not save. Exit!")
        exit()
    else:
        A, b, = hull_simplified.get_halfspaces()
        np.savez(os.path.join(db.get_contact_data_dir(), new_contact_id + ".npz"), A=A, b=b)
        new_contact_profile = {
            'id': new_contact_id,
            'N_vertices': hull_simplified.get_vertices().shape[0],
            "N_faces": hull_simplified.get_halfspaces()[0].shape[0],
            "volume": hull_simplified.compute_volume(),
            'description': contact_profile['description'],
            'attached_to_robot': contact_profile['attached_to_robot'],
            'attached_to_manipulator': new_manip,
            'strategy': args['strategy'],
            'position': [0, 0, 0],
            'orientation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'constraint_coeffs_file': new_contact_id + ".npz",
            'generated_from': contact_profile['id'],
            'generated_on': str(datetime.now())
        }
        if args['strategy'] == "strategy10":
            new_contact_profile['arg_N_samples'] = args['N_samples']
            new_contact_profile['arg_N_vertices'] = args['N_vertices']
        db.insert_profile(new_contact_profile, "contact")
        print("New profile saved.")
    cmd = raw_input("Store constrain coefficients for the input constraint? [y/N]")
    if cmd != "y":
        print("Exit without saving!")
        exit()
    else:
        A, b = hull.get_halfspaces()
        np.savez(os.path.join(db.get_contact_data_dir(), contact_profile['id'] + ".npz"), A=A, b=b)
        contact_profile['constraint_coeffs_file'] = contact_profile['id'] + ".npz"
        contact_profile['N_vertices'] = hull.get_vertices().shape[0]
        contact_profile['N_faces'] = hull.get_halfspaces()[0].shape[0]
        contact_profile['volume'] = hull.compute_volume()
        db.insert_profile(contact_profile, "contact")
        print("Profile inserted. Exit!")

