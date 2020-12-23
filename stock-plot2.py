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

learning_rate_gh = 1e-6
gh_epoch = 1000
gh_patience = 20
batch_size = 32
recur_win_gh = 3

learning_rate_p = 1e-6
p_epoch_factor = 5
recur_win_p = 3

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
win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(noisy_data.train_data, noisy_data.train_t, recur_win_p)
print(win_y.shape, np.sum(win_y**2))
# denom = np.sum(win_y**2)

# directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}'.format('FTSE', 20, 5, 5, 16)
directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_cw0'.format('FTSE', 20, 3, 3, 0)
# directory = '/home/liuwei/Cluster/Stock/{}_p{}_win{}{}_{}'.format('Nikki', 20, 9, 9, 2)
# iter_ = 0
iter_ = 21
data_ = np.load(directory + '/iter{}.npz'.format(iter_))
g = data_['g']
h = data_['h']
noisy_data.train_data = data_['P']

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
    # print('{}'.format(np.sum((predict_one_euler[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2)))
    # print('{}'.format(np.sum((predict_GBM[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2)))
# ##########################################
# denom = 1
denom_test = np.sum(noisy_data.test_data**2)
# denom_test = 1
print(denom_test)

log = open(directory + '/train.log', 'r').readlines()
pos = 0
L1_list = []
L2_list = []
test_list = []
P_list = []
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
for line in log[5:]:
    pos += 1
    if line.startswith('Iter'):
        pos = 0
        continue
    line = line.strip().split()
    if pos == 1:
        # print(line)
        L1_list.append(float(line[1]))
    elif pos == 2:
        print(line)
        L2_list.append(float(line[8][:-1]))
        P_list.append(float(line[12][:-1]))
    elif pos == 3:
        line = [float(i) for i in line[-5:]]
        t1.append(line[0])
        t2.append(line[1])
        t3.append(line[2])
        t4.append(line[3])
        t5.append(line[4])
        # print(line)
        test_list.append(sum(line) / denom_test)
    else:
        continue

print(test_list)
iter_ = np.arange(len(test_list)) + 1
L1_list = L1_list[:len(test_list)]
L2_list = L2_list[:len(test_list)]
# print(t1[89], t2[89], t3[89], t4[89], t5[89])

# plt.figure(figsize=[24, 18])
plt.figure(figsize=[24, 36])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
ax = plt.subplot(3, 3, 1)
plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(iter_, L1_list[:], 'k-', linewidth=3, label='$L_{gh}$')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.4, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(3, 3, 2)
plt.text(-0.1, 1.10, 'B', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(iter_, L2_list[:], 'k-', linewidth=3, label='$L_P$')
plt.tick_params(direction='in', width=3, length=6)
plt.legend(loc='upper left', bbox_to_anchor=[0.4, 0.92], ncol=1)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.010, 0.019, 0.002))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_P}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(3, 3, 3)
plt.text(-0.1, 1.10, 'C', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(x, [1000 * i for i in ep_list[:p]], 'k-', linewidth=3, label='$E_{P}$')
plt.plot(iter_, test_list[:], 'k-', linewidth=3, label='$L_{test}$')

# plt.ylim(0.0014, 0.0031)
# plt.yticks(np.arange(0.0015, 0.0031, 0.0005))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# ax.set_yscale('log')
plt.legend(loc='upper left', bbox_to_anchor=[0.4, 0.92], ncol=1)
# ax.text(.5, .9, '$E_{P}(1\textbf{e-}3)}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{test}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(3, 3, 4)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(x, g, 'k-', linewidth=3, label='final $\hat{g}$')
plt.plot(x[21:74], g[21:74], 'b-', linewidth=3, label='final $\hat{g}$')
# plt.plot(x, noisy_pxt[0, 0, :] / 10)
# plt.plot(x, noisy_pxt[10, 0, :] / 10)
# plt.plot(x, noisy_pxt[12, 0, :] / 10)
# plt.plot(x, noisy_pxt[21, 0, :] / 10)
# plt.plot(x, noisy_pxt[22, 0, :] / 10)
# plt.plot(x, noisy_pxt[25, 0, :] / 10)
# plt.plot(x, noisy_pxt[26, 0, :] / 10)
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.1, 0.92], ncol=1)
# plt.xticks(np.arange(-0.010, 0.010, 0.003))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{g}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

ax = plt.subplot(3, 3, 5)
plt.text(-0.1, 1.10, 'E', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x[21:74], h[21:74], 'k-', linewidth=3, label='final $\hat{h}$')
# plt.plot(x, h, 'r-', linewidth=3, label='final $\hat{h}$')
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.10, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{h}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

ax = plt.subplot(3, 3, 6)
plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot([1, 2, 3, 4, 5], [0.0553, 0.1012, 0.1261, 0.2000, 0.2334], 'b-o', linewidth=3, ms=10, label='$DRN$')
plt.errorbar([1, 2, 3, 4, 5], [0.0733, 0.1064, 0.1374, 0.1533, 0.1770],
             [0.00361, 0.00484, 0.00499, 0.00495, 0.00564], elinewidth=3, label='$FPE NN$')
# plt.errorbar([1, 2, 3, 4, 5], [0.0547, 0.1030, 0.1403, 0.1720, 0.19633],
#              elinewidth=5, label='$FPE NN$')
plt.tick_params(direction='in', width=3, length=6)
plt.legend(loc='upper left', bbox_to_anchor=[0.05, 0.95], ncol=1)
# plt.ylim(0.03, 0.17)
plt.yticks(np.arange(0.04, 0.17, 0.04))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('Sum of squared error ', fontweight='bold')
plt.xlabel('No. of days ahead',  fontweight='bold')

ax = plt.subplot(3, 3, 7)
plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(iter_, P_list[:], 'k-', linewidth=3, label='P difference')
# plt.plot(x, h, 'r-', linewidth=3, label='final $\hat{h}$')
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.10, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{h}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

ax = plt.subplot(3, 3, 8)
plt.text(-0.1, 1.10, 'C', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(x, [1000 * i for i in ep_list[:p]], 'k-', linewidth=3, label='$E_{P}$')
plt.plot(iter_, t1, linewidth=3, label='$L_{test1}$')
plt.plot(iter_, t2, linewidth=3, label='$L_{test2}$')
plt.plot(iter_, t3, linewidth=3, label='$L_{test3}$')
plt.plot(iter_, t4, linewidth=3, label='$L_{test4}$')
plt.plot(iter_, t5, linewidth=3, label='$L_{test5}$')
# plt.ylim(0.0014, 0.0031)
# plt.yticks(np.arange(0.0015, 0.0031, 0.0005))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# ax.set_yscale('log')
plt.legend(loc='upper left', bbox_to_anchor=[0.4, 0.92], ncol=1)
# ax.text(.5, .9, '$E_{P}(1\textbf{e-}3)}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{test}}$', fontweight='bold')
plt.xlabel('iter', fontweight='bold')
plt.show()
