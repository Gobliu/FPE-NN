import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG


# rc('text', usetex=True)
font = {'size': 18}
plt.rc('font', **font)
plt.rc('axes', linewidth=2)
legend_properties = {'weight': 'bold'}

process = 'Boltz'
run_id = 2018
run_ = 0
sigma = 0.02
recur_win_gh = 5
recur_win_p = 5
p_patience = 10
seed = 19822012


directory = '/home/liuwei/Cluster/{}/id{}_p{}_win{}{}_{}'.format(process, run_id, p_patience, recur_win_gh,
                                                                 recur_win_p, run_)
# directory = '/home/liuwei/GitHub/Result/{}/id{}_p{}_win{}{}_{}'.format(process, run_id, p_patience, recur_win_gh,
#                                                                        recur_win_p, run_)
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
print(win_y.shape, np.sum(win_y**2) / 4500)
denom = np.sum(win_y**2) / 4500
denom_test = np.sum(true_pxt[:, :, -5:]**2)

print(denom, denom_test)
# sys.exit()

pos = 0
L1_list = []
L2_list = []
test_list = []
ep_list = []
g_list = []
h_list = []
for line in log[7:]:
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
        L2_list.append(float(line[-3][:-1]) / denom)
    elif pos == 6:
        ep_list.append(float(line[6][:-1]))
    elif pos == 7:
        line = [float(i) for i in line[-5:]]
        test_list.append(sum(line) / denom_test)
    else:
        continue

# print(g_list)
# print(h_list)
# print(ep_list)
# print(test_list)
sum_list = [a + b for a, b in zip(L1_list, L2_list)]
idx = sum_list.index(min(sum_list))
idx = 83
print(idx)
print(g_list[idx], h_list[idx])
print(ep_list[idx])
print(L1_list[idx], L2_list[idx])
print(test_list[idx])
# print(L2_list)
sys.exit()
