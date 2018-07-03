import cdd
import numpy as np
import transport


fmax = 10.0
mu = 0.3

eq_coeffs = [[0, -1, 0, 2, 0, -1, 0]]

ineq_coeffs = [
    [fmax, -1, 0,  0, 0,  0, 0],
    [0,     1, 0,  0, 0,  0, 0],
    [fmax,  0, 0, -1, 0,  0, 0],
    [0,     0, 0,  1, 0,  0, 0],
    [fmax,  0, 0,  0, 0, -1, 0],
    [0,     0, 0,  0, 0,  1, 0],
    [0,    mu,-1,  0, 0,  0, 0],
    [0,    mu, 1,  0, 0,  0, 0],
    [0,     0, 0, mu,-1,  0, 0],
    [0,     0, 0, mu, 1,  0, 0],
    [0,     0, 0,  0, 0, mu,-1],
    [0,     0, 0,  0, 0, mu, 1],
]

A = np.array([[1e-2, 0, 0, 0, -1e-2, 0],
              [1, 0, 1, 0, 1, 0],
              [0.05, 1, 0, 1, -0.05, 1]])
b = np.array([0, -10.0, 0])


def main():
    mat = cdd.Matrix(ineq_coeffs, number_type='float')
    mat.extend(eq_coeffs, linear=True)
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    print(ext)

    ext_np = np.array(ext)[:, 1:]
    print("F's extreme points: \n {:}".format(ext_np))

    ext_wrench = np.dot(ext_np, A.T) + b

    hull = transport.poly_contact.ConvexHull(ext_wrench)
    print("Compute the convex hull of the extrem points: \n "
          "Vertices: \n{:}\n".format(
          hull.get_vertices()))
    print hull.get_halfspaces()

if __name__ == '__main__':
    main()
