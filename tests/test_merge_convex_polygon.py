import pytest
import numpy as np
from transport.plannar_geometry import merge_convex_polygon


def to_float(C):
    return (np.array(C[0]), np.array(C[1]))


@pytest.fixture(params=[1, 2])
def case(request):
    if request.param == 1:
        C1 = ((), (1, 1, -1))
        C2 = ((-1, 1, -0.5), ())
        Cout_exp = ((-1, 1, -0.5), (1, 1, -1))
    if request.param == 2:
        C1 = ((), ([1, 1, -1], [1, 0, -1]))

        C2 = ((-1, 1, -0.5), ())
        Cout_exp = ((-1, 1, -0.5),
                    ([1, 1, -1], [1, 0, -1]))
    return (C1, C2), Cout_exp


@pytest.mark.skip(reason="Currently merging convex polygon is not used. "
                         "This functionality is originally developed to perform constraint pruning."
                         "However, this function is found to be too slow to be useful when the number"
                         "of constraints is too high.")
def test_merge_convex_polygon(case):
    (C1, C2), Cout_exp = case
    C1 = to_float(C1)
    C2 = to_float(C2)
    Cout_exp = to_float(Cout_exp)
    Cout = merge_convex_polygon(C1, C2)
    assert Cout == Cout_exp
