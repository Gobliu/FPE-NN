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
t_gap = 1

learning_rate_gh = 1e-6
gh_epoch = 1000
gh_patience = 20
batch_size = 32
recur_win_gh = 5

learning_rate_p = 1e-6
p_epoch_factor = 5
recur_win_p = 5

valid_win = 5
verb = 2

n_iter = 5000
iter_patience = 20
test_range = 5
sf_range = 7
t_sro = 7
# n_sequence = 40
t_point = 30
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
    # for sample in range(1):
        rep_p = copy.copy(data.train_data[sample, -1, :])
        # print(rep_p[50:60])
        for t_idx in range(predict_t_points):
            k1 = np.matmul(g * rep_p, dx) + np.matmul(h * rep_p, dxx)
            k1.reshape(-1, 1)
            delta_p = k1 * t_gap
            rep_p += delta_p
            # print(rep_p[50:60])
            predict_pxt_euler[sample, t_idx] = copy.copy(rep_p)
    return predict_pxt_euler


F_frag = [[0, 30], [30, 60], [94, 124], [124, 154], [154, 184], [189, 219], [219, 249], [286, 316], [316, 346],
          [389, 419], [419, 449], [449, 479], [497, 527], [533, 563], [563, 593], [691, 721], [721, 751],
          [751, 781], [781, 811], [811, 841], [841, 871], [908, 938], [938, 968], [990, 1020], [1042, 1072],
          [1074, 1104], [1108, 1138], [1184, 1214], [1214, 1244], [1264, 1294],
          [1402, 1432], [1432, 1462], [1462, 1492], [1492, 1522], [1524, 1554], [1564, 1594], [1594, 1624],
          [1624, 1654], [1654, 1684], [1684, 1714], [1714, 1744], [1770, 1800], [1800, 1830], [1845, 1875],
          [1875, 1905], [1905, 1935], [1935, 1965], [1965, 1995], [1995, 2025], [2025, 2055], [2137, 2167],
          [2167, 2197], [2214, 2244], [2244, 2274], [2313, 2343]]

N_frag = [[14, 44], [44, 74], [74, 104], [111, 141], [141, 171], [171, 201], [201, 231], [231, 261], [278, 308],
          [308, 338], [338, 368], [389, 419], [419, 449], [514, 544], [544, 574], [574, 604], [604, 634], [665, 695],
          [728, 758], [769, 799], [799, 829], [829, 859], [859, 889], [889, 919], [931, 961], [961, 991], [991, 1021],
          [1107, 1137], [1137, 1167], [1167, 1197], [1197, 1227], [1236, 1266], [1266, 1296], [1347, 1377],
          [1503, 1533], [1533, 1563], [1563, 1593], [1593, 1623], [1623, 1653], [1653, 1683], [1683, 1713],
          [1713, 1743], [1820, 1850], [1855, 1885], [1976, 2006], [2006, 2036], [2040, 2070], [2096, 2126],
          [2135, 2165], [2165, 2195], [2195, 2225], [2225, 2255], [2317, 2347]]

data_ = np.loadtxt('./stock/data_x.dat')
# data_ *= 100
n_sequence = len(N_frag)
print(n_sequence)
noisy_pxt = np.zeros((n_sequence, t_point, x_points))
for s in range(n_sequence):
    # print(FTSE_fragment[s], FTSE_fragment[s][1] - FTSE_fragment[s][0])
    start, end = N_frag[s]
    noisy_pxt[s, :, :] = data_[start: end, 200:]

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
directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_v3'.format('DOW', 20, 5, 5, 0)
# directory = '/home/liuwei/Cluster/Stock/{}_p{}_win{}{}_{}_v3'.format('Nikki', 20, 5, 5, 0)
iter_ = 0
iter_ = 33
data_ = np.load(directory + '/iter{}.npz'.format(iter_))
g = data_['g'] * t_gap
h = data_['h'] * t_gap
noisy_data.train_data = data_['P']
noisy_data.test_data = data_['test']
predict = data_['predict']
# print(g)
# g = g * t_gap
# print(g)
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

# print(noisy_data.train_data[0, -1, 50:60])
# print('~~~~~~~~~~~~~~~~~~~~')
predict_step_euler = test_step_euler(x, g, h, noisy_data)
# print('~~~~~~~~~~~~~~~~~~~~')
# print(predict_step_euler[0, :, 50:60])
# print(noisy_data.train_data[0, -1, 50:60])
# sys.exit()
predict_GBM = test_one_euler(x, mu*x, sigma*h**2, noisy_data)
for pos in range(test_range):
    error_one_euler = np.sum((predict_one_euler[:, pos, :] - noisy_data.test_data[:, pos, :])**2, axis=1)
    error_GBM = np.sum((predict[:, pos, :] - noisy_data.test_data[:, pos, :])**2, axis=1)
    error_fix = np.sum((noisy_data.train_data[:, -1, :] - noisy_data.test_data[:, pos, :])**2, axis=1)
    error_step_euler = np.sum((predict_step_euler[:, pos, :] - noisy_data.test_data[:, pos, :])**2, axis=1)
    print(pos, error_one_euler[:10])
    print(np.sum(error_one_euler), np.mean(error_one_euler), np.std(error_one_euler), error_one_euler.shape)
    print(np.sum(error_GBM), np.mean(error_GBM), np.std(error_GBM), error_GBM.shape)
    print(np.sum(error_step_euler), np.mean(error_step_euler), np.std(error_step_euler), error_step_euler.shape)
    print(np.sum(error_fix), np.mean(error_fix), np.std(error_fix), error_fix.shape)
    # print(np.mean(error_one_euler ** 2, axis=0))
    # print(np.mean(error_GBM ** 2, axis=0))
    # print(np.std(error_one_euler), np.std(error_GBM))
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
        # print(line)
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
ax = plt.subplot(2, 3, 1)
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

ax = plt.subplot(2, 3, 2)
plt.text(-0.1, 1.10, 'B', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(iter_, L2_list[:], 'k-', linewidth=3, label='$L_P$')
plt.tick_params(direction='in', width=3, length=6)
plt.legend(loc='upper left', bbox_to_anchor=[0.4, 0.92], ncol=1)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.010, 0.019, 0.002))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_P}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 3)
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

ax = plt.subplot(2, 3, 4)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, g, 'k-', linewidth=3, label='final $\hat{g}$')
plt.plot(x[21:74], g[21:74], 'r-', linewidth=3, label='central $\hat{g}$')
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.1, 0.92], ncol=1)
# plt.xticks(np.arange(-0.010, 0.010, 0.003))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{g}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

ax = plt.subplot(2, 3, 5)
plt.text(-0.1, 1.10, 'E', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, h, 'k-', linewidth=3, label='final $\hat{h}$')
plt.plot(x[21:74], h[21:74], 'r-', linewidth=3, label='central $\hat{h}$')
plt.tick_params(direction='in', width=3, length=6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.10, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{\hat{h}}$', fontweight='bold')
plt.xlabel('x',  fontweight='bold')

# [0.00086858 0.00131982 0.00190142 0.00274255 0.00323444] [0.00089647 0.00139888 0.00252809 0.00428139 0.00584501]
# [0.04777163 0.07259029 0.10457818 0.15084014 0.17789439]

# [0.0009372  0.00182643 0.00262831 0.00344672 0.00359603] [0.00066867 0.00168399 0.00332155 0.00349539 0.00387766]
# [0.04967159 0.09680102 0.13930026 0.18267606 0.19058935]

ax = plt.subplot(2, 3, 6)
plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot([1, 2, 3, 4, 5], [0.0435, 0.0760, 0.1139, 0.1327, 0.1554], 'b-o', linewidth=3, ms=10, label='$DRN$')
# ~~~~~~~~~~~~~~~~~~~~~~ FTSE
# plt.errorbar([1, 2, 3, 4, 5], [0.00086858, 0.00131982, 0.00190142, 0.00274255, 0.00323444],
#              [0.00089647, 0.00139888, 0.00252809, 0.00428139, 0.00584501],
#              fmt='o', capthick=5, label='$DRN$')
# plt.errorbar([1, 2, 3, 4, 5], [0.00076, 0.00114, 0.00179, 0.00249, 0.00264],
#              [0.000828, 0.00116, 0.00207, 0.00316, 0.00396],
#              fmt='o', capthick=5, label='$FPE NN$')
# ~~~~~~~~~~~~~~~~~~~~~~ Nikki
plt.errorbar([1, 2, 3, 4, 5], [0.0009372, 0.00182643, 0.00262831, 0.00344672, 0.00359603],
             [0.00066867, 0.00168399, 0.00332155, 0.00349539, 0.00387766],
             fmt='o', capthick=5, label='$DRN$')
plt.errorbar([1, 2, 3, 4, 5], [0.00106, 0.00197, 0.00278, 0.00368, 0.00405],
             [0.00126, 0.00252, 0.00356, 0.00325, 0.00475],
             fmt='o', capthick=5, label='$FPE NN$')
plt.tick_params(direction='in', width=3, length=6)
plt.legend(loc='upper left', bbox_to_anchor=[0.05, 0.95], ncol=1)
# plt.ylim(0.03, 0.17)
# plt.yticks(np.arange(0.04, 0.17, 0.04))
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('Sum of squared error ', fontweight='bold')
plt.xlabel('No. of days ahead',  fontweight='bold')

# ax = plt.subplot(2, 3, 6)
# plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(iter_, P_list[:], 'k-', linewidth=3, label='final $\hat{h}$')
# # plt.plot(x, h, 'r-', linewidth=3, label='final $\hat{h}$')
# plt.tick_params(direction='in', width=3, length=6)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.legend(loc='upper left', bbox_to_anchor=[0.10, 0.92], ncol=1)
# # ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
# plt.ylabel('$\mathbf{\hat{h}}$', fontweight='bold')
# plt.xlabel('x',  fontweight='bold')

plt.show()
