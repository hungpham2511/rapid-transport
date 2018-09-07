import matplotlib.pyplot as plt
import numpy as np
import random
import transport
from scipy.spatial import ConvexHull

N = 100


def gpe_2d(ws_all, ws_sample, N1):
    """
    ws_all: array
        feasible wrench ws_all
    ws_sample: array
        wrench ws_all to cover
    N1: int
        number of vertices
    """
    simp_vertices = []
    # Initially select a random simplex
    random.seed(3)
    simp_samples = random.sample(ws_sample, 3)
    hull = transport.poly_contact.ConvexHull(simp_samples)
    simp_vertices.append(np.array(simp_samples))
    for i in range(N1):
        A, b = hull.get_halfspaces()
        N_faces = len(b)
        i_faces_max = -1
        residue_max = 0
        for j in range(N_faces):
            residue = np.sum(np.where(np.dot(ws_sample, A[j]) - b[j] > 1e-5))  # sum of ws_all not satisfying 
            if residue > residue_max:
                i_faces_max = int(j)
                residue_max = float(residue)

        # All inner ws_all covered
        if i_faces_max == -1:
            break
        else:
            i_max = np.argmax(ws_all.dot(A[i_faces_max]) - b[i_faces_max])
        simp_samples.append(ws_all[i_max])
        hull = transport.poly_contact.ConvexHull(simp_samples)
        simp_vertices.append(np.array(simp_samples))

    return simp_vertices


def get_convexhull_data_2d(pts):
    hull_0 = ConvexHull(pts)
    vlist_0 = list(hull_0.vertices)
    vlist_0.append(hull_0.vertices[0])
    vlist_0 = np.array(vlist_0)
    return pts[vlist_0]


def main():
    # Generate random ws_all
    np.random.seed(1)
    ws_all = np.random.randn(N, 2)
    transform = np.array([[0.9, 0.8], [0, 1.]])
    ws_all = np.array([transform.dot(s) for s in ws_all])
    np.random.seed(2)
    ws_sample = np.random.randn(20, 2) * 0.3 + np.r_[0.5, 0.5]
    ws_all = np.vstack((ws_all, ws_sample))
    # Solve for ws_all
    Y_iters = gpe_2d(ws_all, ws_sample, 10)

    fig = plt.figure(figsize=[3.5, 3.5])
    for i in range(4):
        Yi = Y_iters[i]
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.plot([0, 1, 1, 0], [0, 0, 1, 1], c='gray')
        ax.scatter(ws_all[:, 0], ws_all[:, 1], c='gray', s=10, marker='x')
        ax.scatter(ws_sample[:, 0], ws_sample[:, 1], c='red', s=10)
        Yi_bnd = get_convexhull_data_2d(Yi)
        Ws_bnd = get_convexhull_data_2d(ws_all)
        ax.plot(Yi_bnd[:, 0], Yi_bnd[:, 1], '-^', c='green', markersize=5, lw=2)
        ax.plot(Ws_bnd[:, 0], Ws_bnd[:, 1], '--', c='gray', lw=1)

    plt.tight_layout()
    plt.savefig("gpe.pdf")
    plt.show()

if __name__ == '__main__':
    main()
