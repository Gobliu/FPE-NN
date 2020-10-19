import numpy as np
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG
from NonGridModules.FPLeastSquare_NG import FPLeastSquare_NG
from NonGridModules.FPENet_NG import FPENet_NG
from NonGridModules.Loss import Loss

t_gap = 0.0001
test_range = 5
t_sro = 7
sigma = 0.018
seed = 19822012
data = np.load('/home/liuwei/GitHub/IFPE-Net/Pxt/Bessel/B_f_0_pxt_19822012_sigma0.05.npy')

x = data[0, 0, :]
true_pxt = data[:, 1:, :]

print(data.shape)
# for i in range(100):
#     plt.figure(figsize=[12, 8])
#     plt.plot(x[:], true_pxt[i, 0, :], 'k-', label='p_initial', linewidth=4)
#     plt.plot(x[:], true_pxt[i, -1, :], 'r-', label='p_final', linewidth=4)
#     # plt.plot(x, f_true_pxt[i, 1, :], 'y-', label='f_p_initial', linewidth=4)
#     # plt.plot(x[:], f_true_pxt[i, -1, :], 'g-', label='f_p_final', linewidth=4)
#     # plt.plot(x, f_noisy_pxt[i, 1, :], 'r.', label='p_initial')
#     # plt.plot(x, f_noisy_pxt[i, -1, :], 'b^', label='p_final')
#     plt.legend(fontsize=30)
#     plt.ion()
#     plt.pause(0.5)
#     plt.close()
#     # sys.exit()
#     plt.show()
t = np.zeros((100, 50, 1))
v_ = 0
for i in range(50):
    t[:, i, :] = v_
    v_ += t_gap
print(t[0])

real_g = 1/x - 0.2
real_h = 0.5 * np.ones(x.shape)

lsq = FPLeastSquare_NG(x_coord=x, t_sro=t_sro)

true_data = PxtData_NG(t=t, x=x, data=true_pxt)
true_data.sample_train_split_e2e(test_range=test_range)
t_lsq_g, t_lsq_h, dt, p_mat = lsq.lsq_wo_t(pxt=true_data.train_data, t=true_data.train_t)

plt.figure()
# plt.plot(x, lsq_g, 'r*')
plt.plot(x, t_lsq_g, 'b+')
plt.plot(x, real_g, 'k')
plt.show()

plt.figure()
# plt.plot(x, lsq_h, 'r*')
plt.plot(x, t_lsq_h, 'b+')
plt.plot(x, real_h, 'k')
plt.show()

np.random.seed(seed)
noise = np.random.randn(100, 50, 100)  # normal distribution center N(0, 1) error larger
noisy_pxt = true_pxt + sigma * noise

range_ = 100
print(np.min(noisy_pxt[:, :, :range_]))
# print(f_true_pxt[0, 0, :])
error = true_pxt[:, :, :range_] - noisy_pxt[:, :, :range_]
print(np.sum(error ** 2))
print(np.sum(true_pxt[:, :, :range_] ** 2))
print(np.sum(error ** 2) / np.sum(true_pxt[:, :, :range_] ** 2))
print((np.sum(error ** 2) / np.sum(true_pxt[:, :, :range_] ** 2)) ** 0.5)
np.savez_compressed('./Pxt/Bessel_id{}_{}_sigma{}'.format(2012, seed, sigma), x=x[:range_], t=t,
                    true_pxt=true_pxt[:, :, :range_], noisy_pxt=noisy_pxt[:, :, :range_])
