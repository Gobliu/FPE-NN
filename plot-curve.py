import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG
from NonGridModules.FPLeastSquare_NG import FPLeastSquare_NG
from NonGridModules.FPENet_NG import FPENet_NG
from NonGridModules.Loss import Loss

# rc('text', usetex=True)
font = {'size': 18}
plt.rc('font', **font)
plt.rc('axes', linewidth=2)
legend_properties = {'weight': 'bold'}

process = 'Boltz'
run_id = 1
run_ = 2
sigma = 0.01
recur_win_gh = 13
recur_win_p = 13
p_patience = 10
# seed = 19822012
seed = 821215

# directory = '/home/liuwei/Cluster/{}/id{}_p{}_win{}{}_{}'.format(process, run_id, p_patience, recur_win_gh,
#                                                                  recur_win_p, run_)
directory = '/home/liuwei/GitHub/Result/{}/id{}_p{}_win{}{}_{}'.format(process, run_id, p_patience, recur_win_gh,
                                                                       recur_win_p, run_)
log = open(directory + '/train.log', 'r').readlines()
# log = open('/home/liuwei/GitHub/Result/ghp/Boltz_id2017_p10_win1313_2.txt', 'r').readlines()
data = np.load('./Pxt/{}_id{}_{}_sigma{}.npz'.format(process, run_id, seed, sigma))
x = data['x']
x_points = x.shape[0]
print(x_points)
t = data['t']
# =================
# t = np.zeros((100, 50, 1))
# v_ = 0
# for i in range(50):
#     t[:, i, :] = v_
#     v_ += 0.001
# print(t[0])
# =================
true_pxt = data['true_pxt']
noisy_pxt = data['noisy_pxt']

true_data = PxtData_NG(t=t, x=x, data=true_pxt)
true_data.sample_train_split_e2e(test_range=5)
win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(true_data.train_data, true_data.train_t, recur_win_gh)
print(win_y.shape, np.sum(win_y[0, :, 0]))
denom = np.sum(win_y**2)
denom_test = np.sum(true_pxt[:, :, -5:]**2)

# print(denom, denom_test)
# sys.exit()

pos = 0
L1_list = []
L2_list = []
test_list = []
ep_list = []
g_list = []
h_list = []
gc_list = []
hc_list = []
for line in log[8:]:
    pos += 1
    if line.startswith('Iter'):
        pos = 0
        continue
    line = line.strip().split()
    if pos == 1:
        # print(line)
        L1_list.append(float(line[-1]) / (100**2 * recur_win_p))
    elif pos == 2:
        print(line)
        g_list.append(float(line[-3][:-1])**0.5)
        h_list.append(float(line[-1][:-1])**0.5)
    elif pos == 4:
        # print(line)
        gc_list.append(float(line[-3][:-1])**0.5)
        hc_list.append(float(line[-1][:-1])**0.5)
    elif pos == 6:
        # print(line)
        L2_list.append(float(line[-3][:-1]) / (100**2 * recur_win_p))
    elif pos == 7:
        ep_list.append(float(line[6][:-3]))
    # elif pos == 8:
    #     line = [float(i) for i in line[-5:]]
    #     print(line)
        # test_list.append(sum(line) / (100**2))
    else:
        continue

# print(g_list)
# print(h_list)
# print(gc_list)
# print(hc_list)
# print(L1_list)
# print(L2_list)
# print(ep_list)
# print(test_list)

metric = [x+y for x, y in zip(L1_list, L2_list)]
# print(L1_list)
# print(L2_list)
# print(metric)
print(g_list)
print(gc_list)
print(h_list)
print(hc_list)
# p = metric.index(min(metric[:100]))
# p = 70
p = len(metric) - 1
print(p, len(metric))
print(L1_list[p], L2_list[p], g_list[p], gc_list[p], h_list[p], hc_list[p], ep_list[p])

# sys.exit()
x = np.arange(p) + 1

plt.figure(figsize=[24, 18])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
ax = plt.subplot(2, 3, 4)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, g_list[:p], 'k-', linewidth=3, label='$E_{g}$')
plt.plot(x, gc_list[:p], 'k--', linewidth=3, label=r'$\tilde{E}_{g}$')
# plt.scatter(x1, OU_g, c='r', marker='d', s=50, label='FPE NN')
# # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(-0.00, 0.10, 0.03), fontweight='bold')
# plt.yticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
# plt.yticks(np.arange(-0.05, 0.3, 0.1), fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=[0.25, 0.92], ncol=2, prop={'size': 16})
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('Error of $\mathbf{\hat{g}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 5)
plt.text(-0.1, 1.10, 'E', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, h_list[:p], 'k-', linewidth=3, label='$E_{h}$')
plt.plot(x, hc_list[:p], 'k--', linewidth=3, label=r'$\tilde{E}_{h}$')
# # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
plt.tick_params(direction='in', width=3, length=6)
# # plt.xticks(np.arange(-0.00, 0.10, 0.03), fontweight='bold')
# plt.yticks(fontweight='bold')
# # plt.ylim(1.29, 1.31)
# # plt.ylim(1.2895, 1.3105)
# # plt.yticks(np.arange(1.29, 1.315, 0.01), fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=[0.25, 0.92], ncol=2)
# ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('Error of $\mathbf{\hat{h}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 6)
plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(x, [1000 * i for i in ep_list[:p]], 'k-', linewidth=3, label='$E_{P}$')
plt.plot(x, ep_list[:p], 'k-', linewidth=3, label='$E_{P}$')
# plt.scatter(x2, B_g, c='r', marker='d', s=50, label='FPE NN')
# plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(0.10, 1.11, 0.5), fontweight='bold')
# plt.yticks(fontweight='bold')
# plt.ylim(0.0014, 0.0031)
# plt.yticks(np.arange(0.0015, 0.0031, 0.0005))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# ax.set_yscale('log')
plt.legend(loc='upper left', bbox_to_anchor=[0.35, 0.92], ncol=1)
# ax.text(.5, .9, '$E_{P}(1\textbf{e-}3)}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{E_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 1)
plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, L1_list[:p], 'k-', linewidth=3, label='$L_{gh}$')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(0.10, 1.11, 0.5), fontweight='bold')
# plt.yticks(fontweight='bold')
# plt.ylim(4.995, 5.011)
# plt.yticks(np.arange(5.00, 5.02, 0.01), fontweight='bold')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.35, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{L_1}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 2)
plt.text(-0.1, 1.10, 'B', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, L2_list[:p], 'k-', linewidth=3, label='$L_{P}$')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(0.00, 1.10, 0.5), fontweight='bold')
# plt.ylim(-1.2, 0.5)
# plt.yticks(np.arange(-1.00, 0.6, 0.4), fontweight='bold')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.35, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{L_2}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{P}}$')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 3)
plt.text(-0.1, 1.10, 'C', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot([0, 25, 50, 75, 100, 124], [1.059, 1.012, 1.009, 1.008, 1.007, 1.007], '-o',
             # [1.15, 1.08, 1.03, 1.03, 1.03, 1.03, 1.03],
             linewidth=1, label='$E^1_{test}$')
plt.plot([0, 25, 50, 75, 100, 124], [1.089, 1.011, 1.006, 1.005, 1.004, 1.004], '-^',
             # [1.26, 1.09, 1.00, 1.00, 1.00, 0.99, 0.99],
             linewidth=1, label='$E^2_{test}$')
plt.plot([0, 25, 50, 75, 100, 124], [1.014, 1.007, 1.006, 1.005, 1.004, 1.004], '-d',
             # [1.26, 1.09, 1.00, 1.00, 1.00, 0.99, 0.99],
             linewidth=1, label='$E^3_{test}$')
plt.plot([0, 25, 50, 75, 100, 124], [1.180, 1.022, 1.014, 1.013, 1.014, 1.014], '-s',
             # [1.26, 1.09, 1.00, 1.00, 1.00, 0.99, 0.99],
             linewidth=1, label='$E^4_{test}$')
plt.plot([0, 25, 50, 75, 100, 124], [1.237, 1.036, 1.026, 1.025, 1.024, 1.024], '-*',
             # [1.26, 1.09, 1.00, 1.00, 1.00, 0.99, 0.99],
             linewidth=1, label='$E^5_{test}$')
plt.tick_params(direction='in', width=3, length=6)
plt.xlim(-5, 130)
plt.xticks(np.arange(0, 128, 25))
# plt.yticks(fontweight='bold')
# plt.ylim(-0.2, 2.5)
# plt.yticks(np.arange(0.00, 2.02, 1), fontweight='bold')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(loc='upper left', bbox_to_anchor=[0.2, 0.92], ncol=2)
# ax.text(.5, .9, '$\mathbf{L_{test}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{E_{test}}$')
plt.xlabel('iter',  fontweight='bold')
plt.show()
