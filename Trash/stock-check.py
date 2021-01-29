import sys
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG
from NonGridModules.FPENet_NG import FPENet_NG
from NonGridModules.Loss import Loss

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


def padding_by_axis2_smooth(data, size):
    data_shape = list(data.shape)
    data_shape[2] = int(data_shape[2] + 2 * size)
    data_shape = tuple(data_shape)
    expand = np.zeros(data_shape)
    expand[:, :, size: -size] = data
    expand = signal.savgol_filter(expand, sf_range, 2, axis=2)
    expand = signal.savgol_filter(expand, sf_range, 2, axis=1)
    smooth_data = expand[:, :, size: -size]
    return smooth_data


data_ = np.loadtxt('./stock/data_x.dat')
# data_ *= 100
noisy_pxt = np.zeros((n_sequence, t_point, x_points))
for i in range(n_sequence):
    noisy_pxt[i, :, :] = data_[i * t_point: (i + 1) * t_point, :100]

t = np.zeros((n_sequence, t_point, 1))
for i in range(t_point):
    t[:, i, :] = i * t_gap

smooth_pxt = copy.copy(noisy_pxt)
smooth_pxt[:, :, :-test_range] = padding_by_axis2_smooth(smooth_pxt[:, :, :-test_range], 5)
noisy_data = PxtData_NG(t=t, x=x, data=noisy_pxt)
smooth_data = PxtData_NG(t=t, x=x, data=smooth_pxt)

# end 2 end
noisy_data.sample_train_split_e2e(test_range=test_range)
smooth_data.sample_train_split_e2e(test_range=test_range)
_, _, win_y, _ = PxtData_NG.get_recur_win_e2e(noisy_data.train_data, noisy_data.train_t, 9)
print(win_y.shape, np.sum(win_y**2))
denom = np.sum(win_y**2)

# _, win_t, win_y, win_id = PxtData_NG.get_recur_win_e2e(smooth_data.train_data, smooth_data.train_t, valid_win)
_, win_t, win_y, win_id = PxtData_NG.get_recur_win_e2e(noisy_data.train_data, noisy_data.train_t, valid_win)
# valid_p_p = np.copy(win_x)
valid_p_y = np.copy(win_y)
valid_p_t = np.copy(win_t)

fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)

directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}'.format('Nikki', 20, 5, 5, 3)
# directory = '/home/liuwei/Cluster/Stock/{}_p{}_win{}{}_{}'.format('FTSE', 20, 9, 9, 2)
# iter_ = 285
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

# denom = 1
denom_test = np.sum(noisy_data.test_data**2)
# denom_test = 1
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
        L1_list.append(float(line[1]) / denom)
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
print(noisy_pxt.shape)
pre_P = np.zeros((40, 45, 100))

fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
train_p_x = np.ones((1, x_points, 1))
for i in [0, 50]:
    try:
        data_ = np.load(directory + '/iter{}.npz'.format(i))
        P = data_['P']
        gg_v = data_['g'].reshape((-1, 1, 1))
        hh_v = data_['h'].reshape((-1, 1, 1))
        if i == 0:
            pre_P = np.copy(P)
        print(P.shape)
        print(np.sum((smooth_pxt[:, 44, :] - P[:, -1, :])**2, axis=1))
        print(np.sum((smooth_pxt[1, 44, :] - P[1, -1, :])**2))
        print(np.sum(P), np.sum(noisy_pxt), np.min(P), np.min(noisy_pxt))
        # plt.figure()
        # plt.plot(noisy_pxt[2, 44, :], 'k-', label='noisy', linewidth=1)
        # plt.plot(smooth_pxt[2, 44, :], 'y-', label='smooth', linewidth=1)
        # plt.plot(P[2, 44, :], 'r-', label='trained', linewidth=1)
        # # plt.plot(pre_P[1, -1, :], 'b-', label='p_initial', linewidth=1)
        # plt.legend()
        # plt.title('iter {}'.format(i))
        # plt.show()

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(P, noisy_data.train_t, valid_win)
        train_p_p_ng = np.copy(win_x)
        valid_gh_t = np.copy(win_t)  # key??

        p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                                  fix_g=gg_v, fix_h=hh_v)

        n_sample = train_p_p_ng.shape[0]
        total_train_p_loss_after = 0
        for sample in range(n_sample):
            sample_id, t_id = win_id[sample]  # no true data, end2end
            print('Training P, Sample id: {}, time id {}'.format(sample_id, t_id))
            p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p_ng[sample].reshape(-1, 1, 1)])

            temp = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            temp[temp < 0] = 0
            # update_data.train_data[sample_id, t_id] = temp / np.sum(temp)
            p_loss = p_nn_ng.evaluate([train_p_x, valid_p_t[sample:sample + 1, ...]],
                                      valid_p_y[sample:sample + 1, ...])
            total_train_p_loss_after += p_loss
            if sample == 44:
                one_p_loss = p_loss
                one_pred = p_nn_ng.predict([train_p_x, valid_p_t[sample:sample + 1, ...]])
                one_sum = np.sum((one_pred - valid_p_y[sample:sample + 1, ...])**2, axis=1)
                one_valid = valid_p_y[sample:sample + 1, ...]
                one_t = valid_p_t[sample:sample+1]
                print(one_pred.shape, valid_p_y[sample:sample + 1, ...].shape)
                # sys.exit()
                plt.figure()
                plt.plot(valid_p_y[sample, :, 2], 'k-', label='target', linewidth=1)
                # plt.plot(one_pred[sample, :, 2], 'r-', label='pred', linewidth=1)
                plt.plot(train_p_p_ng[sample], 'b-', label='input', linewidth=1)
                # plt.plot(P[2, 44, :], 'r-', label='trained', linewidth=1)
                # plt.plot(pre_P[1, -1, :], 'b-', label='p_initial', linewidth=1)
                plt.legend()
                plt.title('iter {}'.format(i))
                plt.show()

                np.save(directory + '/iter{}_s{}_pred.npy'.format(i, sample), one_pred)
                np.save(directory + '/iter{}_s{}_valid.npy'.format(i, sample), valid_p_y[sample:sample+1, ...])
        print('loss', total_train_p_loss_after, one_p_loss, one_sum, one_t)
    except:
        # print('no npz', i)
        continue
    # np.save(directory + '/iter{}_s20_pred.npy'.format(i), one_pred)
    # np.save(directory + '/iter{}_s20_valid.npy'.format(i), valid_p_y[20:21, ...])
    print(np.max(one_pred[0, :, -1]), np.max(one_valid[0, :, -1]))
    # sys.exit()
# plt.figure(figsize=[24, 6])
# plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
# ax = plt.subplot(1, 3, 1)
# plt.text(-0.1, 1.10, 'D', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(iter_, L1_list[:], 'k-', linewidth=3, label='$L_gh$')
# # plt.plot(x, gc_list[:p], 'k--', linewidth=3, label=r'$\tilde{E}_{g}$')
# # plt.scatter(x1, OU_g, c='r', marker='d', s=50, label='FPE NN')
# # # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
# plt.tick_params(direction='in', width=3, length=6)
# # plt.xticks(np.arange(-0.00, 0.10, 0.03), fontweight='bold')
# # plt.yticks(fontweight='bold')
# # # plt.ylim(-0.1, 0.4)
# # plt.yticks(np.arange(-0.05, 0.3, 0.1), fontweight='bold')
# plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.92], ncol=1)
# # ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
# plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
# plt.xlabel('iter',  fontweight='bold')
#
# ax = plt.subplot(1, 3, 2)
# plt.text(-0.1, 1.10, 'E', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# plt.plot(iter_, L2_list[:], 'k-', linewidth=3, label='$L_P$')
# # plt.plot(x, hc_list[:p], 'k--', linewidth=3, label=r'$\tilde{E}_{h}$')
# # # plt.scatter(b_x, b_lsq_g, c='r', marker='d', s=100, label='LLS')
# plt.tick_params(direction='in', width=3, length=6)
# plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.92], ncol=1)
# # ax.text(.5, .9, '$\mathbf{h_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
# plt.ylabel('$\mathbf{L_P}$', fontweight='bold')
# plt.xlabel('iter',  fontweight='bold')
#
# ax = plt.subplot(1, 3, 3)
# plt.text(-0.1, 1.10, 'F', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
# # plt.plot(x, [1000 * i for i in ep_list[:p]], 'k-', linewidth=3, label='$E_{P}$')
# plt.plot(iter_, test_list[:], 'k-', linewidth=3, label='$L_{test}$')
# # plt.plot(iter_, 4960 * np.ones(len(test_list)) / denom_test, 'r-', linewidth=3, label='$control$')
# # plt.plot(iter_, 37252 * np.ones(len(test_list)) / denom_test, 'b-', linewidth=3, label='$E_{P}$')
# # plt.ylim(0.0014, 0.0031)
# # plt.yticks(np.arange(0.0015, 0.0031, 0.0005))
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# # ax.set_yscale('log')
# plt.legend(loc='upper left', bbox_to_anchor=[0.65, 0.92], ncol=1)
# # ax.text(.5, .9, '$E_{P}(1\textbf{e-}3)}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
# plt.ylabel('$\mathbf{L_{test}}$', fontweight='bold')
# plt.xlabel('iter',  fontweight='bold')
# plt.show()
