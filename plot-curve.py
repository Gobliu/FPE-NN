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

process = 'OU'
run_id = 2016
run_ = 2
sigma = 0.16
recur_win_gh = 13
recur_win_p = 13
p_patience = 10
seed = 19822012


# directory = '/home/liuwei/GitHub/Result/{}/id{}_p{}_win{}{}_{}'.format(process, run_id, p_patience, recur_win_gh,
#                                                                        recur_win_p, run_)
# log = open(directory + '/train.log', 'r').readlines()
# log = open('/home/liuwei/GitHub/Result/ghp/Boltz_id2017_p10_win1313_2.txt', 'r').readlines()
data = np.load('./Pxt/{}_id{}_{}_sigma{}.npz'.format(process, run_id, seed, sigma))
x = data['x']
x_points = x.shape[0]
print(x_points)
# t = data['t']
# =================
t = np.zeros((100, 50, 1))
v_ = 0
for i in range(50):
    t[:, i, :] = v_
    v_ += 0.001
print(t[0])
# =================
true_pxt = data['true_pxt']
noisy_pxt = data['noisy_pxt']

true_data = PxtData_NG(t=t, x=x, data=true_pxt)
true_data.sample_train_split_e2e(test_range=5)
win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(true_data.train_data, true_data.train_t, 13)
print(win_y.shape, np.sum(win_y**2) / 4500)
denom = np.sum(win_y**2) / 4500
denom_test = np.sum(true_pxt[:, :, -5:]**2)

print(denom, denom_test)
sys.exit()

pos = 0
L1_list = []
L2_list = []
test_list = []
ep_list = []
g_list = []
h_list = []
for line in log[8:]:
    pos += 1
    if line.startswith('Iter'):
        pos = 0
        continue
    line = line.strip().split()
    if pos == 1:
        L1_list.append(float(line[-1]) / denom)
    elif pos == 2:
        g_list.append(float(line[-3][:-1]))
        h_list.append(float(line[-1][:-1]))
    elif pos == 5:
        L2_list.append(float(line[-1]) / denom)
    elif pos == 6:
        ep_list.append(float(line[6][:-1]))
    elif pos == 8:
        line = [float(i) for i in line[-5:]]
        test_list.append(sum(line) / denom_test)
    else:
        continue

# print(g_list)
# print(h_list)
# print(ep_list)
# print(test_list)
# sys.exit()

x = np.arange(len(L1_list))

plt.figure(figsize=[24, 18])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
ax = plt.subplot(2, 3, 1)
plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, g_list, 'k-', linewidth=3, label='$g_{error}$')
# plt.scatter(x1, OU_g, c='r', marker='d', s=50, label='FPE NN')
# # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(-0.00, 0.10, 0.03), fontweight='bold')
# plt.yticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
# plt.yticks(np.arange(-0.05, 0.3, 0.1), fontweight='bold')
# plt.legend(loc='upper left', bbox_to_anchor=[0.05, 0.85], ncol=1)
# ax.text(.5, .9, 'OU', horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
plt.ylabel('$\mathbf{g_{error}}$')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 2)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, h_list, 'k-', linewidth=3, label='$h_{error}$')
# # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
plt.tick_params(direction='in', width=3, length=6)
# # plt.xticks(np.arange(-0.00, 0.10, 0.03), fontweight='bold')
# plt.yticks(fontweight='bold')
# # plt.ylim(1.29, 1.31)
# # plt.ylim(1.2895, 1.3105)
# # plt.yticks(np.arange(1.29, 1.315, 0.01), fontweight='bold')
# plt.legend(loc='upper left', bbox_to_anchor=[0.05, 0.85], ncol=1)
# ax.text(.5, .9, 'OU', horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
plt.ylabel('$\mathbf{h_{error}}$')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 3)
plt.text(-0.1, 1.10, 'C', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, [100 * i for i in ep_list], 'k-', linewidth=3, label='$\epsilon$')
# plt.scatter(x2, B_g, c='r', marker='d', s=50, label='FPE NN')
# plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
# plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(0.10, 1.11, 0.5), fontweight='bold')
# plt.yticks(fontweight='bold')
# ax.set_yscale('log')
# plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.85], ncol=1)
# ax.text(.5, .9, 'Bessel', horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
plt.ylabel('$\mathbf{ \epsilon (1E-2)}$')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 4)
plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, L1_list, 'k-', linewidth=3, label='$L_1$')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(0.10, 1.11, 0.5), fontweight='bold')
# plt.yticks(fontweight='bold')
# plt.ylim(4.995, 5.011)
# plt.yticks(np.arange(5.00, 5.02, 0.01), fontweight='bold')
# plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.85], ncol=1)
# ax.text(.5, .9, 'Bessel', horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
plt.ylabel('$\mathbf{L_1}$')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 5)
plt.text(-0.1, 1.10, 'E', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, L2_list, 'k-', linewidth=3, label='$L_2$')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(0.00, 1.10, 0.5), fontweight='bold')
# plt.ylim(-1.2, 0.5)
# plt.yticks(np.arange(-1.00, 0.6, 0.4), fontweight='bold')
# plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.85], ncol=1)
# ax.text(.5, .9, 'Wealth', horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
plt.ylabel('$\mathbf{L_2}$')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 6)
plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x, test_list, 'k-', linewidth=3, label='$L_{test}$')
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(np.arange(0.00, 1.10, 0.5), fontweight='bold')
# plt.yticks(fontweight='bold')
# plt.ylim(-0.2, 2.5)
# plt.yticks(np.arange(0.00, 2.02, 1), fontweight='bold')
# plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.85], ncol=1)
# ax.text(.5, .9, 'Wealth', horizontalalignment='center', transform=ax.transAxes, fontweight='bold')
plt.ylabel('$\mathbf{L_{test}}$')
plt.xlabel('iter',  fontweight='bold')
plt.show()
