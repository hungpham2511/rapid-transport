""" This scripts compare several solvers when subjecting to the optimization programs
that occur when using TOPPRA.

"""
from toppra_app.plannar_geometry import line_intersect, halfplanes_intersection
import numpy as np
import cvxpy as cvx
import time
import mosek

from qpoases import (PyOptions as Options, PyPrintLevel as PrintLevel,
                     PyReturnValue as ReturnValue, PySQProblem as SQProblem)

from copy import copy
import cdd

inf = 0.0


def prune_halfspaces_allstage(a_vec, b_vec, c_vec):
    """

    Notes
    -----
    This function assumes the constraint x >= 0.

    Parameters
    ----------
    a_vec
    b_vec
    c_vec

    Returns
    -------

    """
    N = a_vec.shape[0]
    take_indices = set()
    for i in range(N):
        take_indices_ = halfplanes_intersection(a_vec[i], b_vec[i], c_vec[i])
        take_indices = take_indices.union(set(take_indices_))
    take_indices = list(take_indices)
    take_indices.sort()
    return take_indices


def prune_input(input_, method='allstage'):
    if method == 'allstage':
        N = input_['N']
        take_indices = prune_halfspaces_allstage(input_['a'][:N], input_['b'][:N], input_['c'][:N])
        input_new = input_.copy()  # need to copy, so that original input is unchanged
        input_new['a'] = np.ascontiguousarray(input_['a'][:N, take_indices])
        input_new['b'] = np.ascontiguousarray(input_['b'][:N, take_indices])
        input_new['c'] = np.ascontiguousarray(input_['c'][:N, take_indices])
    if method == 'stagewise':
        input_new = input_.copy()
        a_new = []
        b_new = []
        c_new = []
        for i in range(input_['N']):
            take_indices = halfplanes_intersection(input_['a'][i], input_['b'][i], input_['c'][i])
            a_new.append(input_['a'][i, take_indices])
            b_new.append(input_['b'][i, take_indices])
            c_new.append(input_['c'][i, take_indices])
        input_new['a'] = a_new
        input_new['b'] = b_new
        input_new['c'] = c_new
    return input_new


def solve_with_qpOASES(input_, strategy):
    print "Start solving with {:}, warmstart: {:}, prune: {:}".format(
        'qpOASES', strategy['warmstart'], strategy['prune'])
    if strategy['warmstart']:
        return _solve_with_qpOASES_warm(input_, strategy)
    else:
        return _solve_with_qpOASES_cold(input_, strategy)


def solve_with_mosek(input_, strategy):
    print "Start solving with {:}, warmstart: {:}, prune: {:}".format(
        'mosek', strategy['warmstart'], strategy['prune'])
    if strategy['warmstart']:
        return _solve_with_mosek_warm(input_, strategy)
    else:
        return _solve_with_mosek_cold(input_, strategy)


def _solve_with_qpOASES_warm(input_, strategy):
    output_ = {'opt_vars': [], 'solve_times': [], 'solver_name': 'qpOASES',
               'prune': strategy['prune'],
               'warmstart': True}
    t0 = time.time()
    if strategy['prune']:
        input_ = prune_input(input_, method='allstage')
    t_prune = time.time() - t0
    output_['t_prune'] = t_prune

    option = Options()
    option.printLevel = PrintLevel.NONE
    nV = 2
    l = np.zeros(2)
    h = np.zeros(2)
    ux = np.zeros(2)

    nC = len(input_['a'][0])
    solver = SQProblem(nV, nC)
    solver.setOptions(option)
    H_zeros = np.zeros((2, 2))
    lAinf = - np.ones(nC) * 10000000
    A = np.zeros((nC, nV))
    hA = np.zeros(nC)

    for i in range(input_['N']):
        t0 = time.time()

        l[0] = input_['lu'][i]
        l[1] = input_['lx'][i]
        h[0] = input_['hu'][i]
        h[1] = input_['hx'][i]
        A[:, 0] = input_['a'][i]
        A[:, 1] = input_['b'][i]
        hA[:] = - input_['c'][i]
        if i == 0:
            res = solver.init(H_zeros, input_['g'][i], A, l, h, lAinf, hA, np.array([1000]))
        else:
            res = solver.hotstart(H_zeros, input_['g'][i], A, l, h, lAinf, hA, np.array([1000]))
        if res == ReturnValue.SUCCESSFUL_RETURN:
            solver.getPrimalSolution(ux)
        else:
            ux = [-123, -123]
        output_['opt_vars'].append(np.array(ux))
        output_['solve_times'].append(time.time() - t0)
    return output_

def _solve_with_qpOASES_cold(input_, strategy):
    output_ = {'opt_vars': [], 'solve_times': [], 'solver_name': 'qpOASES',
               'prune': strategy['prune'],
               'warmstart': False}
    t0 = time.time()
    if strategy['prune']:
        input_ = prune_input(input_, method='stagewise')
    t_prune = time.time() - t0
    output_['t_prune'] = t_prune

    option = Options()
    option.printLevel = PrintLevel.NONE
    nV = 2
    l = np.zeros(2)
    h = np.zeros(2)
    ux = np.zeros(2)

    for i in range(input_['N']):
        t0 = time.time()

        nC = len(input_['a'][i])
        solver = SQProblem(nV, nC)
        solver.setOptions(option)
        H_zeros = np.zeros((2, 2))
        lAinf = - np.ones(nC) * 10000000
        A = np.zeros((nC, nV))
        hA = np.zeros(nC)

        l[0] = input_['lu'][i]
        l[1] = input_['lx'][i]
        h[0] = input_['hu'][i]
        h[1] = input_['hx'][i]
        A[:, 0] = input_['a'][i]
        A[:, 1] = input_['b'][i]
        hA[:] = - input_['c'][i]
        res = solver.init(H_zeros, input_['g'][i], A, l, h, lAinf, hA, np.array([1000]))
        if res == ReturnValue.SUCCESSFUL_RETURN:
            solver.getPrimalSolution(ux)
        else:
            ux = [-123, -123]
        output_['opt_vars'].append(np.array(ux))
        output_['solve_times'].append(time.time() - t0)
    return output_


def _solve_with_mosek_cold(input_, strategy):
    output_ = {'opt_vars': [], 'solve_times': [], 'solver_name': 'mosek',
               'prune': strategy['prune'],
               'warmstart': False}
    t0 = time.time()
    if strategy['prune']:
        input_ = prune_input(input_, method='stagewise')
    t_prune = time.time() - t0
    output_['t_prune'] = t_prune
    for i in range(input_['N']):
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                t0 = time.time()
                m = input_['a'][i].shape[0]
                task.appendcons(m)
                task.appendvars(2)
                task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.free)
                task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
                # print(i)
                bkx = [mosek.boundkey.ra, mosek.boundkey.ra]
                blx = [input_['lu'][i], input_['lx'][i]]
                bux = [input_['hu'][i], input_['hx'][i]]
                for j in [0, 1]:
                    task.putcj(j, input_['g'][i][j])
                    task.putvarbound(j, bkx[j], blx[j], bux[j])
                task.putacol(0, range(m), input_['a'][i])
                task.putacol(1, range(m), input_['b'][i])
                task.putconboundslice(0, m, [mosek.boundkey.up] * m, [-inf] * m, - input_['c'][i])
                res = task.optimize()
                if res == mosek.rescode.ok:
                    xx = [0, 0.]
                    task.getxx(mosek.soltype.bas, xx)
                else:
                    xx = [-123, -123]  # Special number of infeasible instances
                output_['opt_vars'].append(xx)
                output_['solve_times'].append(time.time() - t0)

    return output_


def _solve_with_mosek_warm(input_, strategy):
    output_ = {'opt_vars': [], 'solve_times': [], 'solver_name': 'mosek',
               'prune': strategy['prune'], 'warmstart': True }
    t0 = time.time()
    if strategy['prune']:
        input_ = prune_input(input_, method='allstage')
    output_['t_prune'] = time.time() - t0
    m = input_['a'].shape[1]
    env = mosek.Env()
    task = env.Task(0, 0)
    task.appendcons(m)
    task.appendvars(2)
    task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.free)
    task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.free_simplex)
    for i in range(input_['N']):
        # print(i)
        t0 = time.time()
        bkx = [mosek.boundkey.ra, mosek.boundkey.ra]
        blx = [input_['lu'][i], input_['lx'][i]]
        bux = [input_['hu'][i], input_['hx'][i]]
        for j in [0, 1]:
            task.putcj(j, input_['g'][i][j])
            task.putvarbound(j, bkx[j], blx[j], bux[j])
        task.putacol(0, range(m), input_['a'][i])
        task.putacol(1, range(m), input_['b'][i])
        task.putconboundslice(0, m, [mosek.boundkey.up] * m, [-inf] * m, - input_['c'][i])
        res = task.optimize()
        if res == mosek.rescode.ok:
            xx = [0, 0.]
            task.getxx(mosek.soltype.bas, xx)
        else:
            xx = [-123, -123]  # Special number of infeasible instances
        output_['opt_vars'].append(xx)
        output_['solve_times'].append(time.time() - t0)
    task.__del__()
    env.__del__()
    return output_


def solve_with_cvxpy(input_, strategy, solver='ECOS'):
    if strategy['warmstart']:
        return None
    output_ = {'opt_vars': [], 'solve_times': [], 'solver_name': '{:}_cvxpy'.format(solver)}
    print "Start solving with {:}".format(output_['solver_name'])
    for i in range(input_['N']):
        # print(i)
        t0 = time.time()
        ux = cvx.Variable(2)
        constraints = [
            ux[0] * input_['a'][i] + ux[1] * input_['b'][i] + input_['c'][i] <= 0,
            ux[0] >= input_['lu'][i], ux[0] <= input_['hu'][i],
            ux[1] >= input_['lx'][i], ux[1] <= input_['hx'][i],
        ]
        objective = cvx.Minimize(input_['g'][i] * ux)
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=solver)
        output_['opt_vars'].append(ux.value)
        output_['solve_times'].append(time.time() - t0)
    return output_


def load_test_data(select=None):
    f_ = np.load('tests/object_transport_test.npz')
    a_vec = f_['a_vec']
    b_vec = f_['b_vec']
    c_vec = f_['c_vec']

    inputs = []

    # Case 0
    m = a_vec.shape[0]
    input_ = {'a': a_vec, 'b': b_vec, 'c': c_vec, 'N': 20,
              'lx': np.zeros(m), 'hx': np.ones(m),
              'lu': -np.ones(m) * 10, 'hu': np.ones(m) * 10,
              'g': np.array([(0.2, -1) for i in range(m)]),
              'set_no': 1, 'description': 'Large test case, m approx 250k'}
    inputs.append(input_)

    # Case 1
    m = a_vec.shape[0]
    input_ = {'a': a_vec, 'b': b_vec, 'c': c_vec, 'N': 20,
              'lx': np.ones(m) * 0.2, 'hx': np.ones(m) * 0.2,
              'lu': -np.ones(m) * 10, 'hu': np.ones(m) * 10,
              'g': np.array([(1.0, 1) for i in range(m)]),
              'set_no': 3, 'description': 'Large test case, m approx 250k, x has equality constraints'}
    inputs.append(input_)

    # Case 2
    m = 100
    input_ = {'a': a_vec[:, :m], 'b': b_vec[:, :m], 'c': c_vec[:, :m], 'N': 50,
              'lx': np.zeros(m), 'hx': np.ones(m) * 3,
              'lu': -np.ones(m) * 100, 'hu': np.ones(m) * 100,
              'g': np.array([(0.3, 1) for i in range(m)]),
              'set_no': 2, 'description': 'nil'}
    inputs.append(input_)

    # Case 3
    m = 2000
    input_ = {'a': a_vec[:, :m], 'b': b_vec[:, :m], 'c': c_vec[:, :m], 'N': 50,
              'lx': np.zeros(m), 'hx': np.ones(m) * 5,
              'lu': -np.ones(m) * 100, 'hu': np.ones(m) * 100,
              'g': np.array([(0.3, 1) for i in range(m)]),
              'set_no': 2, 'description': 'nil'}
    inputs.append(input_)

    # Case 4
    m = 5000
    input_ = {'a': a_vec[:, :m], 'b': b_vec[:, :m], 'c': c_vec[:, :m], 'N': 50,
              'lx': np.zeros(m), 'hx': np.ones(m),
              'lu': -np.ones(m), 'hu': np.ones(m),
              'g': np.array([(0.3, 1) for i in range(m)]),
              'set_no': 2, 'description': 'nil'}
    inputs.append(input_)

    # Case 5
    m = 20000
    input_ = {'a': a_vec[:, :m], 'b': b_vec[:, :m], 'c': c_vec[:, :m], 'N': 50,
              'lx': np.zeros(m), 'hx': np.ones(m),
              'lu': -np.ones(m), 'hu': np.ones(m),
              'g': np.array([(0.3, 1) for i in range(m)]),
              'set_no': 4, 'description': 'Small test case, m={:d}'.format(m)}
    inputs.append(input_)

    for i, input_ in enumerate(inputs):
        input_['set_no'] = i + 1
    if select is None:
        return inputs
    else:
        return [inputs[select]]


def report(input_, outputs):
    string = "-" * 50 + '\n'
    string += "          Comparison for input set {:d}\n".format(input_['set_no'])
    string += "-" * 50 + '\n'
    string += "    Number of stages: {:d}\n".format(input_['N'])
    string += "    Number of constraints per stage: {:d}\n".format(input_['a'].shape[1])
    string += "    Description: {:}\n\n".format(input_['description'])
    for output_ in outputs:
        string += "Solver: {:}\n".format(output_['solver_name'])
        string += "Settings\n"
        string += "    Warmstart: {:}\n".format(output_['warmstart'])
        string += "    Prune: {:}\n".format(output_['prune'])
        string += "Timing\n"
        string += "    Total Prune time: {:f}\n".format(output_['t_prune'])
        string += "    Average Prune time: {:f}\n".format(output_['t_prune'] / input_['N'])
        string += "    Total solve time: {:.5f} seconds\n".format(sum(output_['solve_times']))
        string += "    Average solve time: {:.5f} seconds\n".format(sum(output_['solve_times']) / input_['N'])
        string += "    Max solve time: {:.5f} seconds\n\n".format(max(*output_['solve_times']))

    # Check for solution consistency
    if len(outputs) == 1:
        string += "Solution consistency check not performed."
    else:
        string += "Solution consistency\n"
        for i in range(len(outputs)):
            string += "    |sol{:d} - sol{:d}| = {:f}\n".format(
                i, 0, np.linalg.norm(
                    np.array(outputs[i]['opt_vars']).flatten() - np.array(outputs[0]['opt_vars']).flatten()
                ))
    string += "\n----------------------------------------------\n"

    return string


if __name__ == '__main__':
    inputs = load_test_data(0)
    solve_with_cvxpy_ECOS = lambda input_: solve_with_cvxpy(input_, "ECOS")
    solve_with_cvxpy_mosek = lambda input_: solve_with_cvxpy(input_, "MOSEK")
    solve_with_cvxpy_glpk = lambda input_: solve_with_cvxpy(input_, "GLPK")
    solver_totest = [solve_with_mosek, solve_with_qpOASES]
    all_strategies = [
        {'warmstart': False, 'prune': False},
        # {'warmstart': False, 'prune': True},
        {'warmstart': True, 'prune': False},
        # {'warmstart': True, 'prune': True}
    ]

    reports = []
    for input_ in inputs:
        outputs = []
        for solver in solver_totest:
            for strategy_ in all_strategies:
                output_ = solver(input_, strategy_)
                outputs.append(output_)
        reports.append(report(input_, outputs))

    output_file = '/home/hung/.temp/compare_solvers_report'
    with open(output_file, 'a') as f:
        from datetime import datetime
        f.write(str(datetime.now()) + "\n")
        for r in reports:
            f.write(r)

