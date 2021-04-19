import numpy as np

from scipy import stats


# data = np.load('./Pxt/Boltz_id{}_{}_sigma{}_200.npz'.format(1, 821215, 0.01))
# data = np.load('./Pxt/Bessel_id{}_{}_sigma{}.npz'.format(12, 19822012, 0.01))
# data = np.load('./Pxt/OU_id{}_{}_sigma{}_200.npz'.format(2017, 19822012, 0.16))
data = np.load('./Pxt/Tri_id{}_{}_sigma{}.npz'.format(1, 19822012, 0.01))
x = data['x']
true_pxt = data['true_pxt']
noisy_pxt = data['noisy_pxt']

# data = np.load('/home/liuwei/GitHub/IFPE-Net/Pxt/Bessel/B_f_10_pxt_19822012_sigma0.05.npy')
# data = np.load('/home/liuwei/GitHub/IFPE-Net/Pxt/Bessel/B_f_10_noisy_19822012_sigma0.05.npy')
# x = data[0, 0, :]
# noisy_pxt = data[:, 1:, :]
# print(data.shape)
# print(x.shape, noisy_pxt.shape)

################################################ stock
# x = np.linspace(-0.015, 0.015, num=100, endpoint=True)
# data_ = np.loadtxt('./stock/data_x.dat')
# # data_ *= 100
# noisy_pxt = np.zeros((40, 50, 100))
# for i in range(40):
#     noisy_pxt[i, :, :] = data_[i * 50: (i + 1) * 50, 200:]
######################################################3

bar = np.max(noisy_pxt) * 0.01
for pos in range(x.shape[0]):
    p = noisy_pxt[:, :, pos].reshape(-1, 1)
    print(pos, stats.ttest_1samp(p, bar))

# print(t_list)
print(np.max(noisy_pxt))

