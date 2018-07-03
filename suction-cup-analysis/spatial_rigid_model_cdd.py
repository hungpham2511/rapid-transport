import cdd
import numpy as np
import time
import transport
import matplotlib.pyplot as plt


PA = 15.0  # Newton
mu = 0.25
r = 10.5e-3  # radius of the cup
N = 6
Delta = 1e-3
nvars = 3 * N + 3  # 3 comps for each contactforce and 3 for suction force
alphas = [2 * np.pi * i / N for i in range(N)]
l = 0  # distance from {O}

fmax = 100.0 / N


def main():
    # Equality A_eq f_bar == 0
    A_eq = np.zeros((3, nvars))
    b_eq = np.zeros(3)
    A_eq[:3, N * 3: N * 3 + 3] = np.eye(3)
    b_eq[:3] = [0, 0, -PA]

    # Inequality A_ineq f_bar <= b_ineq
    A_ineq = np.zeros((4 * N + 2 * N, nvars))
    b_ineq = np.zeros(4 * N + 2 * N)
    
    for i in range(N):
        # inner approximation of Colomb friction constraint
        A_ineq[4 * i: 4 * i + 4, 3 * i: 3 * i + 3] = [[-1, -1, -mu],
                                                      [-1, 1, -mu],
                                                      [1, 1, -mu],
                                                      [1, -1, -mu]]
        # max bound on vertical component forces
        A_ineq[4 * N + 2 * i: 4 * N + 2 * i + 2, 3 * i: 3 * i + 3] = [
            [0, 0, -1],
            [0, 0, 1]
        ]
        b_ineq[4 * N + 2 * i: 4 * N + 2 * i + 2] = [0, fmax]

    # H-rep to V-rep
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


    # Transform to wrench space: w_O = F f
    F = np.zeros((6, 3 * N + 3))
    for i in range(N + 1):
        F[3:, 3 * i: 3 * i + 3] = np.eye(3)
    for i in range(N):
        F[:3, 3 * i: 3 * i + 3] = [[0, -l, - r * np.cos(alphas[i])],
                                   [l, 0, r * np.sin(alphas[i])],
                                   [r * np.cos(alphas[i]), - r * np.sin(alphas[i]), 0]]
    F[:3, 3 * N: 3 * N + 3] = [[0, -l, 0],
                               [l, 0, 0],
                               [0, 0, 0]]
    w0_extreme_pts = f_extreme_pts.dot(F.T)

    # convex hull, then 
    transport.utils.preview_plot([(w0_extreme_pts, 'o', {'markersize': 5})], dur=100)
    w0_hull = transport.poly_contact.ConvexHull(w0_extreme_pts)
    print("Computed convex hull has {0:d} vertices and {1:d} faces".format(
        len(w0_hull.get_vertices()), w0_hull.get_halfspaces()[1].shape[0]
    ))

    # save coefficients
    A, b = w0_hull.get_halfspaces()
    cmd = raw_input("Save constraint coefficients A, b y/[N]?")
    if cmd == "y":
        np.savez("/home/hung/Dropbox/ros_data/toppra_application/bigcup_analytical_rigid3.npz", A=A, b=b)
        print("Saved coefficients to database!")
    else:
        exit("abc")


if __name__ == '__main__':
    main()
