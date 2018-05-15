from transport.plannar_geometry import halfplanes_intersection
import numpy as np
import cvxpy as cvx
import pytest


@pytest.fixture(params=[1, 2])
def fixture(request):
    if request.param == 1:
        N_random = 100
        np.random.seed(1)
        seeds = np.random.randint(1000000, size=3)
        np.random.seed(seeds[0])
        a = np.random.randn(N_random)
        np.random.seed(seeds[1])
        b = np.random.randn(N_random)
        np.random.seed(seeds[2])
        c = np.random.rand(N_random) - 1 - 1e-3

        xlow = 0
        xhigh = 1.0
        ulow = -1
        uhigh = 1

    if request.param == 2:
        N_random = 1000
        np.random.seed(2)
        seeds = np.random.randint(1000000, size=3)
        np.random.seed(seeds[0])
        a = np.random.randn(N_random)
        np.random.seed(seeds[1])
        b = np.random.randn(N_random)
        np.random.seed(seeds[2])
        c = np.random.rand(N_random) - 1 - 1e-3

        xlow = 0
        xhigh = 100
        ulow = -10000
        uhigh = 100

    edges = halfplanes_intersection(a, b, c, xlow, xhigh, ulow, uhigh)
    return a, b, c, edges, ulow, uhigh, xlow, xhigh

def test_random_shooting(fixture):
    """ If the original halfspaces and the pruned one are consistent, then
    the optimal values of two linear programs subject to the two sets of
    constraints must equal.
    """
    a, b, c, edges, ulow, uhigh, xlow, xhigh = fixture

    # %% Solve with cvxpy
    for i in range(10):
        d_random = np.random.randn(2)

        # Original problem
        y = cvx.Variable(2)
        u = y[0]
        x = y[1]
        constraint = [a * u + b * x + c <= 0,
                    x >= xlow, x <= xhigh, u <= uhigh, ulow <= u]
        obj = cvx.Minimize(d_random * y)
        prob = cvx.Problem(obj, constraint)
        prob.solve()
        val1 = prob.value
        # Reduced problem
        if len(edges) == 0:
            continue
        y = cvx.Variable(2)
        u = y[0]
        x = y[1]
        constraint = [a[edges] * u + b[edges] * x + c[edges] <= 0,
                      x >= xlow, x <= xhigh, u <= uhigh, ulow <= u]
        obj = cvx.Minimize(d_random * y)
        prob = cvx.Problem(obj, constraint)
        prob.solve()
        val2 = prob.value

        np.testing.assert_allclose(val1, val2, atol=1e-8)
