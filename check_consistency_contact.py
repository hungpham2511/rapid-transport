import toppra_app, os
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

db = toppra_app.database.Database()
contact_profile_original = db.retrieve_profile("nonprehensile_gripper", "contact")
contact_profile_simp = db.retrieve_profile("nonprehensile_gripper_simplified_strategy9a", "contact")

file_ = np.load(os.path.join(db.get_contact_data_dir(), contact_profile_original['constraint_coeffs_file']))
A_orig, b_orig = file_['A'], file_['b']

file_ = np.load(os.path.join(db.get_contact_data_dir(), contact_profile_simp['constraint_coeffs_file']))
A_simp, b_simp = file_['A'], file_['b']

centroid = np.r_[-1.11627846, -0.01857521, -0.01109492, -0.1352319,   7.06683072,  0.62000883]

print A_orig.shape, A_simp.shape

# %%
diffs = []
for i in range(1000):
    unit_vec = toppra_app.utils.generate_random_unit_vectors(6, 1)[0]

    d = cvx.Variable()
    constraints_orig = [A_orig * (centroid + d * unit_vec) <= b_orig]
    constraints_simp = [A_simp * (centroid + d * unit_vec) <= b_simp]
    prob = cvx.Problem(cvx.Maximize(d), constraints_orig)
    prob.solve(solver="MOSEK")
    d_orig = prob.value
    prob = cvx.Problem(cvx.Maximize(d), constraints_simp)
    prob.solve(solver="MOSEK")
    d_simp = prob.value

    diffs.append(d_orig - d_simp)




