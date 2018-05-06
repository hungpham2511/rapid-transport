import toppra_app, os
import numpy as np

contact = 'suctioncup_kindlebox_467795874a'
torque_safety_factor = 4

db = toppra_app.database.Database()

contact_profile = db.retrieve_profile(contact, "contact")

file_ = np.load(os.path.join(db.get_contact_data_dir(), contact_profile['id'] + ".npz"))
A = file_['A']
b = file_['b']

T_ft_suction = np.array([[  1.22464680e-16,  -1.00000000e+00,   2.22044605e-16, 1.62092562e-17],
                         [  2.22044605e-16,  -2.22044605e-16,  -1.00000000e+00, -7.30000000e-02],
                         [  1.00000000e+00,   1.22464680e-16,   2.22044605e-16, 3.37500000e-02],
                         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])

Ad_T_ft_suction = toppra_app.utils.Ad(T_ft_suction)

# A_suc, b_suc
# A inv(Ad_T_ft_suc.T) <= b

b_suc = np.array(b)
A_suc = A.dot(np.linalg.inv(Ad_T_ft_suction.T))

A_suc[:, 0] = A_suc[:, 0] * torque_safety_factor
A_suc[:, 1] = A_suc[:, 1] * torque_safety_factor

# convert back to ft frame
A_new = A_suc.dot(Ad_T_ft_suction.T)
b_new = np.copy(b_suc)

np.savez(os.path.join(db.get_contact_data_dir(), contact_profile['id'] + "_safe.npz"),
         A=A_new, b=b_new)


