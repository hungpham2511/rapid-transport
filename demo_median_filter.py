import numpy as np
import toppra_app, os
import matplotlib.pyplot as plt
import openravepy as orpy
import scipy.signal

VERBOSE = True
if VERBOSE:
    import coloredlogs
    coloredlogs.install(level='DEBUG')
    np.set_printoptions(3)

db = toppra_app.database.Database()
contact_profile = db.retrieve_profile("suctioncup_kindlebox2", "contact")
# contact_profile = db.retrieve_profile("suctioncup", "contact")
object_profile = db.retrieve_profile("small_alum_block_2", "object")
robot_profile = db.retrieve_profile("gripper1", "robot")

T_ft_suction = np.array([[  1.22464680e-16,  -1.00000000e+00,   2.22044605e-16, 1.62092562e-17],
                         [  2.22044605e-16,  -2.22044605e-16,  -1.00000000e+00, -7.30000000e-02],
                         [  1.00000000e+00,   1.22464680e-16,   2.22044605e-16, 3.37500000e-02],
                         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])

Ad_T_ft_suction = toppra_app.utils.Ad(T_ft_suction)

ws_all = []
for file_name in contact_profile['raw_data']:
    file_dir = os.path.join(db.get_contact_data_dir(), file_name)
    ws = toppra_app.utils.load_data_ati_log(file_dir)
    ws_all.append(ws)
ws_all_ = np.vstack(ws_all)

ws_all = np.array(ws_all_)
for i in range(ws_all.shape[0]):
    ws_all[i] = Ad_T_ft_suction.T.dot(ws_all_[i])

# ws_all[:, 0] = scipy.signal.medfilt(ws_all[:, 0], 15)
# ws_all[:, 1] = scipy.signal.medfilt(ws_all[:, 1], 15)
# ws_all[:, 2] = scipy.signal.medfilt(ws_all[:, 2], 15)

fig, axs = plt.subplots(2, 1)
axs[0].set_title("Torque")
to_plot = [(0, "taux"), (1, "tauy"), (2, "tauz")]
for i, name in to_plot:
    axs[0].plot(ws_all[:, i], label=name)
axs[0].legend()
to_plot = [(0, "fx"), (1, "fy"), (2, "fz")]
axs[1].set_title("Force")
for i, name in to_plot:
    axs[1].plot(ws_all[:, 3 + i], label=name)
axs[1].legend()
plt.show()


