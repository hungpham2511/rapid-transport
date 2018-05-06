import numpy as np
import toppra_app
import matplotlib.pyplot as plt
import scipy.signal

ws = toppra_app.utils.load_data_ati_log(
    "/home/hung/Dropbox/ros_data/toppra_application/testtraj_vibration")

"""In this test, two experiments were conducted. In the first one, no
object is attached to the suction head. In the second one, the kindle
box is attached.

See below for the trial indices. The sampling rate of the record is
1000Hz.

"""
empty_trial_indices = range(29192, 34027)
with_mass_trial_indices = range(79235, 83928)


T_ft_suction = np.array([[  1.22464680e-16,  -1.00000000e+00,   2.22044605e-16, 1.62092562e-17],
                         [  2.22044605e-16,  -2.22044605e-16,  -1.00000000e+00, -7.30000000e-02],
                         [  1.00000000e+00,   1.22464680e-16,   2.22044605e-16, 3.37500000e-02],
                         [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])

Ad_T_ft_suction = toppra_app.utils.Ad(T_ft_suction)

wrench_empty_trial_ = ws[empty_trial_indices]
wrench_mass_trial_ = ws[with_mass_trial_indices]

#%% FFT
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
A = np.fft.fft(wrench_empty_trial_[:, 3], 1000)
A2 = np.fft.fft(wrench_mass_trial_[:, 3], 1000)
axs[0].plot(np.abs(A), label='empty')
axs[1].plot(np.abs(A2), label='with mass')
for ax in axs:
    ax.legend()
plt.show()


wrench_empty_trial = np.array([Ad_T_ft_suction.T.dot(w) for w in wrench_empty_trial_])
wrench_mass_trial = np.array([Ad_T_ft_suction.T.dot(w) for w in wrench_mass_trial_])

# for i in range(6):
#     wrench_empty_trial[:, i] = scipy.signal.savgol_filter(wrench_empty_trial[:, i], 61, 5)
#     wrench_mass_trial[:, i] = scipy.signal.savgol_filter(wrench_mass_trial[:, i], 61, 5)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

for i, label in enumerate(['x', 'y', 'z']):
    axs[0].plot(np.arange(wrench_mass_trial.shape[0]) / 1000.0, wrench_mass_trial[:, i], label=label)
    axs[1].plot(np.arange(wrench_empty_trial.shape[0]) / 1000.0, wrench_empty_trial[:, i], label=label)
axs[0].legend()
axs[1].legend()
plt.show()

#%% FFT
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
A = np.fft.fft(wrench_empty_trial[:, 0], 1000)
A2 = np.fft.fft(wrench_mass_trial[:, 0], 1000)
axs[0].plot(np.abs(A), label='empty')
axs[1].plot(np.abs(A2), label='with mass')
for ax in axs:
    ax.legend()
plt.show()
