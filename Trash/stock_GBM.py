import sys
import os
import copy

import numpy as np
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG

np.set_printoptions(suppress=True)
font = {'size': 18}
plt.rc('font', **font)
plt.rc('axes', linewidth=2)
legend_properties = {'weight': 'bold'}

name = 'Noisy'
x_min = -0.015
x_max = 0.015
x_points = 100
# x_points = config.X_POINTS
# print(x_gap)
t_gap = 0.001

# learning_rate_gh = 1e-6
# gh_epoch = 1000
# gh_patience = 20
# batch_size = 32
recur_win_gh = 5

# learning_rate_p = 1e-6
# p_epoch_factor = 5
recur_win_p = 5

# valid_win = 9
verb = 2

# n_iter = 5000
# iter_patience = 20
test_range = 5
sf_range = 7
t_sro = 7
n_sequence = 40
t_point = 50
x = np.linspace(x_min, x_max, num=x_points, endpoint=True)


def test_one_euler(x, g, h, data):
    dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
    dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)
    n_sample = data.test_data.shape[0]
    predict_t_points = data.test_data.shape[1]
    predict_pxt_euler = np.zeros((n_sample, predict_t_points, x.shape[0]))
    for sample in range(data.n_sample):
        p0 = data.train_data[sample, -1, :]
        relative_t = data.test_t[sample, :, :] - data.train_t[sample, -1, :]
        k1 = np.matmul(g * p0, dx) + np.matmul(h * p0, dxx)
        k1.reshape(-1, 1)
        relative_t.reshape(1, -1)
        delta_p = np.multiply(k1, relative_t)
        predict_pxt_euler[sample] = p0 + delta_p
    return predict_pxt_euler


data_ = np.loadtxt('./stock/data_x.dat')
# data_ *= 100
noisy_pxt = np.zeros((n_sequence, t_point, x_points))
start_ = 0
for i in range(n_sequence):
    noisy_pxt[i, :, :] = data_[start_ + i * t_point: start_ + (i + 1) * t_point, :100]

# sum_ = np.sum(data_[:1000, :100], axis=0)
# print(sum_.shape)

t = np.zeros((n_sequence, t_point, 1))
for i in range(t_point):
    t[:, i, :] = i * t_gap

noisy_data = PxtData_NG(t=t, x=x, data=noisy_pxt)

# end 2 end
noisy_data.sample_train_split_e2e(test_range=test_range)
win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(noisy_data.train_data, noisy_data.train_t, 5)
print(win_y.shape, np.sum(win_y**2))
denom = np.sum(win_y**2)

directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}'.format('FTSE', 20, 5, 5, 13)
iter_ = 41
# iter_ = 0
data_ = np.load(directory + '/iter{}.npz'.format(iter_))
g = data_['g']
h = data_['h']
train_data = data_['P']
# predict = data_['predict']
# test = data_['test']
print(np.sum((noisy_data.train_data - train_data)**2))
print((np.sum((noisy_data.train_data - train_data)**2) / np.sum(noisy_data.train_data ** 2))**0.5)
noisy_data.train_data = train_data

# ##########################################
# print(np.sum((noisy_data.test_data - test)**2))
mu = np.sum(x[37:74] * g[37:74]) / np.sum(x[37:74]**2)
print(-mu)
# print(np.sum((mu*x[37:74] - g[37:74])**2))
# mu *= 1.000000001
# print(np.sum((mu*x[37:74] - g[37:74])**2))
sigma = np.sum(x[37:74]**2 * h[37:74]) / np.sum(x[37:74]**4)
print(sigma)
# print(np.sum((sigma*x[37:74]**2 - h[37:74])**2))
# sigma *= 1.00000001
# sigma *= 0.999999999
# print(np.sum((sigma*x[37:74]**2 - h[37:74])**2))
# ##########################################
predict_one_euler = test_one_euler(x, g, h, noisy_data)
predict_GBM = test_one_euler(x, mu*x, sigma*h**2, noisy_data)
for pos in range(test_range):
    error_one_euler = predict_one_euler[:, pos, :] - noisy_data.test_data[:, pos, :]
    error_GBM = predict_GBM[:, pos, :] - noisy_data.test_data[:, pos, :]
    print(np.sum(error_one_euler ** 2))
    print(np.sum(error_GBM ** 2))
    print(np.std(error_one_euler), np.std(error_GBM))
# ##########################################

# denom = 1
denom_test = np.sum(noisy_data.test_data**2)
# denom_test = 1
print(denom_test)
print()

plt.figure(figsize=[24, 18])
ax = plt.subplot(2, 3, 1)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, predict_one_euler[0, 1, :], 'b-', linewidth=3, label='final $\hat{g}$')
plt.plot(x, predict_GBM[0, 1, :], 'r-', linewidth=3, label='final $\hat{g}$')
plt.plot(x, noisy_data.test_data[0, -1, :], 'k-', linewidth=3, label='final $\hat{g}$')
# plt.plot(x[21:74], mu * x[21:74], 'r-', linewidth=3, label='final $\hat{g}$')
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.1, 0.92], ncol=1)
# plt.xticks(np.arange(-0.010, 0.010, 0.003))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{g}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

ax = plt.subplot(2, 3, 4)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x[21:74], g[21:74], 'k-', linewidth=3, label='final $\hat{g}$')
# plt.plot(x[21:74], mu * x[21:74], 'r-', linewidth=3, label='final $\hat{g}$')
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.1, 0.92], ncol=1)
# plt.xticks(np.arange(-0.010, 0.010, 0.003))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{g}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

ax = plt.subplot(2, 3, 5)
plt.text(-0.1, 1.10, 'E', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x[21:74], h[21:74], 'k-', linewidth=3, label='final $\hat{h}$')
plt.plot(x[21:74], sigma * x[21:74]**2, 'r-', linewidth=3, label='final $\hat{g}$')
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.10, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{h}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

plt.show()
