import numpy as np


def normal(line):
    return line[0], line[1]


def line_intersect(line1, line2):
    det = line1[1] * line2[0] - line1[0] * line2[1]
    x = (- line1[1] * line2[2] + line1[2] * line2[1]) / det
    y = (line1[0] * line2[2] - line1[2] * line2[0]) / det
    return (x, y)


def highest_ycoord(C):
    if C[0].shape[0] == 0 or C[1].shape[0] == 0:
        return 10e9
    if normal(C[1][0])[1] > 0 and normal(C[0][0])[1] > 0:
        point = line_intersect(C[1][0], C[0][0])
        return point[1]
    return 10e9


def merge_convex_polygon(C1, C2):
    ymost1 = highest_ycoord(C1)
    ymost2 = highest_ycoord(C2)
    event = min(ymost1, ymost2)

    while event != -10e9:
        left_edge_C1
        right_edge_C1
        left_edge_C2
        right_edge_C2



