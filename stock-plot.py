import sys
import os
import copy

import numpy as np
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG

np.set_printoptions(suppress=True)

name = 'Noisy'
x_min = -0.015
x_max = 0.015
x_points = 100
# x_points = config.X_POINTS
# print(x_gap)
t_gap = 0.001

learning_rate_gh = 1e-5
gh_epoch = 1000
gh_patience = 20
batch_size = 32
recur_win_gh = 9

learning_rate_p = 1e-5
p_epoch_factor = 5
recur_win_p = 9

valid_win = 9
verb = 2

n_iter = 5000
iter_patience = 20
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


def test_step_euler(x, g, h, data):
    dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
    dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)
    n_sample = data.test_data.shape[0]
    predict_t_points = data.test_data.shape[1]
    predict_pxt_euler = np.zeros((n_sample, predict_t_points, x.shape[0]))
    for sample in range(data.n_sample):
        rep_p = data.train_data[sample, -1, :]
        for t_idx in range(predict_t_points):
            k1 = np.matmul(g * rep_p, dx) + np.matmul(h * rep_p, dxx)
            k1.reshape(-1, 1)
            delta_p = k1 * t_gap
            rep_p += delta_p
            predict_pxt_euler[sample, t_idx] = copy.copy(rep_p)
    return predict_pxt_euler


data_ = np.loadtxt('./stock/data_x.dat')
data_ *= 100
noisy_pxt = np.zeros((n_sequence, t_point, x_points))
for i in range(n_sequence):
    noisy_pxt[i, :, :] = data_[i * t_point: (i + 1) * t_point, :100]

t = np.zeros((n_sequence, t_point, 1))
for i in range(t_point):
    t[:, i, :] = i * t_gap

noisy_data = PxtData_NG(t=t, x=x, data=noisy_pxt)

# end 2 end
noisy_data.sample_train_split_e2e(test_range=test_range)

directory = '/home/liuwei/GitHub/Result/Stock/p{}_win{}{}_{}'.format(20, 9, 9, 7)
# iter_ = 129
# data_ = np.load(directory + '/iter{}.npz'.format(iter_))
# g = data_['g']
# h = data_['h']
# noisy_data.train_data = data_['P']

# predict_one_euler = test_one_euler(x, g, h, noisy_data)
# for pos in range(test_range):
#     print('{} \t'.format(np.sum((predict_one_euler[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2) /
#                          np.sum(noisy_data.test_data[:, pos, :]) ** 2))
#
# predict_one_euler = test_step_euler(x, g, h, noisy_data)
# for pos in range(test_range):
#     print('{} \t'.format(np.sum((predict_one_euler[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2) /
#                          np.sum(noisy_data.test_data[:, pos, :]) ** 2))

denom = 1
# denom_test = np.sum(noisy_data.test_data**2)
denom_test = 1
print(denom_test)

log = open(directory + '/train.log', 'r').readlines()
pos = 0
L1_list = []
L2_list = []
test_list = []
for line in log[5:]:
    pos += 1
    if line.startswith('Iter'):
        pos = 0
        continue
    line = line.strip().split()
    if pos == 1:
        # print(line)
        L1_list.append(float(line[-1]) / denom)
    elif pos == 2:
        # print(line)
        L2_list.append(float(line[-4][:-1]) / denom)
    elif pos == 3:
        line = [float(i) for i in line[-5:]]
        # print(line)
        test_list.append(sum(line) / denom_test)
    else:
        continue

print(test_list)
iter_ = np.arange(len(test_list)) + 1
L1_list = L1_list[:len(test_list)]
L2_list = L2_list[:len(test_list)]

plt.figure(figsize=[16, 12])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
ax = plt.subplot(1, 3, 1)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(iter_, L1_list[:], 'k-', linewidth=3, label='$L_gh$')
# plt.plot(x, gc_list[:p], 'k--', linewidth=3, label=r'$\tilde{E}_{g}$')
# plt.scatter(x1, OU_g, c='r', marker='d', s=50, label='FPE NN')
# # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(-0.00, 0.10, 0.03), fontweight='bold')
# plt.yticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
# plt.yticks(np.arange(-0.05, 0.3, 0.1), fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(1, 3, 2)
plt.text(-0.1, 1.10, 'E', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(iter_, L2_list[:], 'k-', linewidth=3, label='$L_P$')
# plt.plot(x, hc_list[:p], 'k--', linewidth=3, label=r'$\tilde{E}_{h}$')
# # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
plt.tick_params(direction='in', width=3, length=6)
plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_P}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(1, 3, 3)
plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(x, [1000 * i for i in ep_list[:p]], 'k-', linewidth=3, label='$E_{P}$')
plt.plot(iter_, test_list[:], 'k-', linewidth=3, label='$L_{test}$')
# plt.plot(iter_, 4960 * np.ones(len(test_list)) / denom_test, 'r-', linewidth=3, label='$control$')
# plt.plot(iter_, 37252 * np.ones(len(test_list)) / denom_test, 'b-', linewidth=3, label='$E_{P}$')
# plt.ylim(0.0014, 0.0031)
# plt.yticks(np.arange(0.0015, 0.0031, 0.0005))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# ax.set_yscale('log')
plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.92], ncol=1)
# ax.text(.5, .9, '$E_{P}(1\textbf{e-}3)}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{test}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')
plt.show()
