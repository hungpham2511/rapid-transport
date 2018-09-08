import cdd
import numpy as np
import time, os
import transport
import matplotlib.pyplot as plt
import yaml
import openravepy as orpy
import argparse

transport.utils.setup_logging("DEBUG")

ROBOT_MODEL_DIR = os.path.expanduser('~/git/toppra-object-transport/models/denso_ft_sensor_suction.robot.xml')
CONTACT_OUTPUT_ID = "analytical_rigid" + "1234"

# Parameters of the contact model
PA = 22.0  # (Newton) suction force
mu = 0.3  # coeff of friction
r = 12.5e-3  # radius of the cup
N = 6  # Number of points
nvars = 3 * N + 3  # 3 comps for each contactforce and 3 for suction force
alphas = [2 * np.pi * i / N for i in range(N)]  # angle of the contact points
l = 0  # distance from {O}
fmax = 200.0 / N  # Maximum allowable magnitude of individual contact forces


def main(simplify=False):
    """Compute the contact stability constraint.

    The contact stability is the H-representation of the set of
    feasible interaction wrenches that a suction cup exerts on the
    object.

    There are two main steps. First, find the extreme points in the
    space of concatenated component vectors. Second, find an inner
    approximation of this set using guidance from a dynamic model.
    The second stage is carried out if `simplify` is True, otherwise
    it is neglected.
    """
    # Step 1: find the extreme points in the space of concatenated
    # component vectors.

    # Define equality and inequality constraint:
    #      A_eq f_bar == 0
    #      A_ineq f_bar <= b_ineq
    A_eq = np.zeros((3, nvars))
    b_eq = np.zeros(3)
    A_eq[:3, N * 3: N * 3 + 3] = np.eye(3)
    b_eq[:3] = [0, 0, -PA]

    A_ineq = np.zeros((4 * N + 2 * N, nvars))
    b_ineq = np.zeros(4 * N + 2 * N)
    for i in range(N):
        # inner approximation of Colomb friction constraint
        A_ineq[4 * i: 4 * i + 4, 3 * i: 3 * i + 3] = [[-1, -1, -mu],
                                                      [-1, 1, -mu],
                                                      [1, 1, -mu],
                                                      [1, -1, -mu]]
        # max/min bounds on vertical component forces
        A_ineq[4 * N + 2 * i: 4 * N + 2 * i + 2, 3 * i: 3 * i + 3] = [[0, 0, -1],
                                                                      [0, 0, 1]]
        b_ineq[4 * N + 2 * i: 4 * N + 2 * i + 2] = [0, fmax]

    # Transform from H-rep to V-rep using cdd
    t0 = time.time()
    mat = cdd.Matrix(np.hstack((b_ineq.reshape(-1, 1), -A_ineq)), number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.extend(np.hstack((b_eq.reshape(-1, 1), -A_eq)), linear=True)
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    ext = np.array(ext)
    t_elapsed = time.time() - t0
    print("Approximate with N={2:d} points:\n\tFound {0:d} extreme points in {1:10.3f} secs".format(ext.shape[0], t_elapsed, N))
    f_extreme_pts = ext[:, 1: 1 + 3 * N + 3]

    # Transform to interacting wrench space:
    # w_O = F f, where F is defined below
    F = np.zeros((6, 3 * N + 3))
    for i in range(N):
        F[:3, 3 * i: 3 * i + 3] = [[0, -l, - r * np.cos(alphas[i])],
                                   [l, 0, r * np.sin(alphas[i])],
                                   [r * np.cos(alphas[i]), - r * np.sin(alphas[i]), 0]]
    F[:3, 3 * N: 3 * N + 3] = [[0, -l, 0],
                               [l, 0, 0],
                               [0, 0, 0]]
    for i in range(N + 1):
        F[3:, 3 * i: 3 * i + 3] = np.eye(3)
    w0_extreme_pts = f_extreme_pts.dot(F.T)

    # Step 2: Use a robot model to generate better points.

    # REQUIRED OUTPUT w0_hull: the convex hull of the vertices of the
    # set of physically realization wrenches.
    if simplify:
        env = orpy.Environment()
        env.Load(ROBOT_MODEL_DIR)
        robot = env.GetRobots()[0]
        contact_base = transport.Contact(robot, "denso_suction_cup2",
                                         np.eye(4), None, None, raw_data=w0_extreme_pts)
        solid_object = transport.SolidObject.init_from_dict(robot, {
            'object_profile': "bluenb",
            'object_attach_to': "denso_suction_cup2",
            "T_link_object": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 12.5e-3], [0, 0, 0, 1]],
            "name": "obj"})
        cs = transport.ContactSimplifier(robot, contact_base, solid_object, N_vertices=60)
        contact_simp, w0_hull = cs.simplify()
    else:
        w0_hull = transport.poly_contact.ConvexHull(w0_extreme_pts)

    transport.utils.preview_plot([(w0_extreme_pts, 'o', {'markersize': 5}), (w0_hull.vertices, 'x', {'markersize': 7})],
                                 dur=100)
    print("Computed convex hull has {0:d} vertices and {1:d} faces".format(
        len(w0_hull.get_vertices()), w0_hull.get_halfspaces()[1].shape[0]
    ))

    # save coefficients
    A, b = w0_hull.get_halfspaces()
    contact_profile = {CONTACT_OUTPUT_ID: {
        "id": CONTACT_OUTPUT_ID,
        "attached_to_manipulator": "denso_suction_cup2",
        "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "position": [0, 0, 0],
        "constraint_coeffs_file": CONTACT_OUTPUT_ID + ".npz",
        "params": {"simplify": simplify, "N": N, "PA": PA, "mu": mu, "r": r, "fmax": fmax}
    }}
    print("db entry (to copy manually)\n\nbegin -----------------\n\n{:}\nend--------".format(yaml.dump(contact_profile)))
    cmd = raw_input("Save constraint coefficients A, b y/[N]?")
    if cmd == "y":
        np.savez("{:}.npz".format(CONTACT_OUTPUT_ID), A=A, b=b)
        print("Saved coefficients to {:}!".format(os.path.curdir))
    else:
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(True)
