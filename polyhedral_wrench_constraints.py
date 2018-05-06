"""
NOTE:  This script is not used.

Apparently, the procedure described in this script is both
computationally challenging and useless.

It is computationally challenging because for 3 connection points, each exerts a wrench
in R6, the set of feasible net wrenches is a polyhedron with 30k vertices.

Even worse, for 4 connection points, the script does not even terminate. Clearly the
procedure is not scalable. The algorithm for projecting polyhedron has, if I remember
well, theoretically exponential complexity.

It is useless because the large amount of vertices leads to an
equivalently large number of faces, from which high computational
cost follows.

For both reasons above, I have decided to abandon this approach. Instead, I will
experimentally find out the set of feasible wrenches using the FT sensor.

This script computes the constraints which define the set of feasible net wrenches
on the origin of a frame fixed to a solid object with m interaction wrenches.

It outputs numpy arrays (F, b) that define the polyheral constraints

F w <= b.

Input format:

- The first line contains n, k1, ..., kn, where n is the number of
  contacts, ki is the number of constraints.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


from pymanoid.polyhedra import project_polytope

input_string = """
4 10 10 10 10

10.0 10.0 -5.0
1 0 0 0 0 0 0.5
0 1 0 0 0 0 0.5
0 0 1 0 0 0 0.5
-1 0 0 0 0 0 0.5
0 -1 0 0 0 0 0.5
0 0 -1 0 0 0 0.5
0 0 0 0 0 1 20
0 0 0 0 0 -1 -1
0 0 0 1 0 -0.2 0
0 0 0 -1 0 -0.2 0

10.0 -10.0 -5.0
1 0 0 0 0 0 0.5
0 1 0 0 0 0 0.5
0 0 1 0 0 0 0.5
-1 0 0 0 0 0 0.5
0 -1 0 0 0 0 0.5
0 0 -1 0 0 0 0.5
0 0 0 0 0 1 20
0 0 0 0 0 -1 -1
0 0 0 1 0 -0.2 0
0 0 0 -1 0 -0.2 0

-10.0 10.0 -5.0
1 0 0 0 0 0 0.5
0 1 0 0 0 0 0.5
0 0 1 0 0 0 0.5
-1 0 0 0 0 0 0.5
0 -1 0 0 0 0 0.5
0 0 -1 0 0 0 0.5
0 0 0 0 0 1 20
0 0 0 0 0 -1 -1
0 0 0 1 0 -0.2 0
0 0 0 -1 0 -0.2 0

-10.0 -10.0 -5.0
1 0 0 0 0 0 0.5
0 1 0 0 0 0 0.5
0 0 1 0 0 0 0.5
-1 0 0 0 0 0 0.5
0 -1 0 0 0 0 0.5
0 0 -1 0 0 0 0.5
0 0 0 0 0 1 20
0 0 0 0 0 -1 -1
0 0 0 1 0 -0.2 0
0 0 0 -1 0 -0.2 0

"""

input_string = """
3 10 10 10

10.0 10.0 -5.0
1 0 0 0 0 0 0.5
0 1 0 0 0 0 0.5
0 0 1 0 0 0 0.5
-1 0 0 0 0 0 0.5
0 -1 0 0 0 0 0.5
0 0 -1 0 0 0 0.5
0 0 0 0 0 1 20
0 0 0 0 0 -1 -1
0 0 0 1 0 -0.2 0
0 0 0 -1 0 -0.2 0

10.0 -10.0 -5.0
1 0 0 0 0 0 0.5
0 1 0 0 0 0 0.5
0 0 1 0 0 0 0.5
-1 0 0 0 0 0 0.5
0 -1 0 0 0 0 0.5
0 0 -1 0 0 0 0.5
0 0 0 0 0 1 20
0 0 0 0 0 -1 -1
0 0 0 1 0 -0.2 0
0 0 0 -1 0 -0.2 0


-10.0 -10.0 -5.0
1 0 0 0 0 0 0.5
0 1 0 0 0 0 0.5
0 0 1 0 0 0 0.5
-1 0 0 0 0 0 0.5
0 -1 0 0 0 0 0.5
0 0 -1 0 0 0 0.5
0 0 0 0 0 1 20
0 0 0 0 0 -1 -1
0 0 0 1 0 -0.2 0
0 0 0 -1 0 -0.2 0

"""

def process_string_input(input_string):
    """

    Parameters
    ----------
    input_string: string
        Input string, which contains information on the connection wrench constraints.

    Returns
    -------
    positions: array
    F_array: list of array
    b_array: array

    """
    input_lines = filter(lambda s: s != "", input_string.splitlines())  # Remove empty lines
    first_line = input_lines.pop(0)
    m = np.fromstring(first_line, sep=" ", dtype=int)[0]
    nb_constraints = np.fromstring(first_line, sep=" ", dtype=int)[1:]

    assert m == len(nb_constraints)

    positions = []
    F_array = []
    b_array = []
    for i in range(m):
        index = i + sum(nb_constraints[:i])
        positions.append(np.fromstring(input_lines[index], sep=" ", dtype=float))
        F = []
        b = []
        for k in range(nb_constraints[i]):
            index = i + sum(nb_constraints[:i]) + k + 1
            line_data = np.fromstring(input_lines[index], sep=" ", dtype=float)
            F.append(line_data[:-1])
            b.append(line_data[-1])
        F_array.append(np.array(F))
        b_array.append(np.array(b))
    b_array = np.array(b_array)

    return positions, F_array, b_array

def skew(p):
    """

    Parameters
    ----------
    p: array
        (3,) array.

    Returns
    -------
    out: array
        (3, 3) array.

    """
    return np.array([[0.0, -p[2], p[1]],
                     [p[2], 0, -p[0]],
                     [-p[1], p[0], 0]
                     ])

if __name__ == "__main__":
    positions, F_array, b_array = process_string_input(input_string)
    F_block = scipy.linalg.block_diag(*F_array)
    b_block = np.hstack(b_array)

    T_list = []
    for p in positions:
        T = np.eye(6)
        T[:3, 3:] = skew(p)
        T_list.append(T)

    T_block = np.hstack((T_list))

    m = len(positions) * 6

    result = project_polytope((F_block, b_block), (np.zeros((1, m)), np.array([0])), (T_block, np.zeros(6)))

    print result



