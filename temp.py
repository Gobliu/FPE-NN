import sys
import os

import numpy as np
from scipy import signal
from keras import callbacks, backend, losses
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG
from NonGridModules.FPLeastSquare_NG import FPLeastSquare_NG
from NonGridModules.FPENet_NG import FPENet_NG
from NonGridModules.Loss import Loss

from GridModules.GaussianSmooth import GaussianSmooth

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


def test_steps(x, g, h, data):
    dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
    dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)
    n_sample = data.test_data.shape[0]
    predict_t_points = data.test_data.shape[1]
    predict_pxt_euler = np.zeros((n_sample, predict_t_points, x.shape[0]))
    for sample in range(data.n_sample):
        p0 = data.train_data[sample, -1, :]
        relative_t = data.test_t[sample, :, :] - data.train_t[sample, -1, :]
        k1 = np.matmul(g * p0, dx) + np.matmul(h * p0, dxx)
        # print(p0.shape, k1.shape, relative_t.shape)
        k1.reshape(-1, 1)
        relative_t.reshape(1, -1)
        # print(p0.shape, k1.shape, relative_t.shape)
        delta_p = np.multiply(k1, relative_t)
        # print(delta_p.shape)
        predict_pxt_euler[sample] = p0 + delta_p
        # print(p0[:5], delta_p[:5, :5], predict_pxt_euler[sample, :5, :5])
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


def main(smooth_gh=0.1, smooth_p=False):
    run_ = 0
    while run_ < 100:
        directory = '/home/liuwei/GitHub/Result/Stock/p{}_win{}{}_{}'.format(gh_patience, recur_win_gh,
                                                                             recur_win_p, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break

    x = np.linspace(x_min, x_max, num=x_points, endpoint=True)

    data = np.loadtxt('./stock/data_x.dat')
    data *= 100
    noisy_pxt = np.zeros((n_sequence, t_point, x_points))
    for i in range(n_sequence):
        noisy_pxt[i, :, :] = data[i * t_point: (i+1) * t_point, :100]

    t = np.zeros((n_sequence, t_point, 1))
    for i in range(t_point):
        t[:, i, :] = i * t_gap

    print(np.sum(noisy_pxt[-1, -1, :]))

    log = open(directory + '/train.log', 'w')
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('t_sro: {} \n'.format(t_sro))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, sf_range))
    smooth_pxt = padding_by_axis2_smooth(noisy_pxt, 5)      # test information leakage
    log.write('Difference of pxt before and after smooth: {} ratio {}\n'.
              format(np.sum((smooth_pxt - noisy_pxt)**2)**0.5,
                     np.sum((smooth_pxt - noisy_pxt)**2)**0.5/np.sum(noisy_pxt**2)**0.5))
    log.close()

    if smooth_p:
        update_pxt = np.copy(smooth_pxt)
    else:
        update_pxt = np.copy(noisy_pxt)

    noisy_data = PxtData_NG(t=t, x=x, data=noisy_pxt)
    smooth_data = PxtData_NG(t=t, x=x, data=smooth_pxt)
    update_data = PxtData_NG(t=t, x=x, data=update_pxt)
    update_data_valid = PxtData_NG(t=t, x=x, data=update_pxt)

    # end 2 end
    noisy_data.sample_train_split_e2e(test_range=test_range)
    smooth_data.sample_train_split_e2e(test_range=test_range)
    update_data.sample_train_split_e2e(test_range=test_range)
    update_data_valid.sample_train_split_e2e(test_range=test_range)

    print(noisy_data.train_data.shape, noisy_data.test_data.shape)
    # sys.exit()

    smooth_data.train_data = padding_by_axis2_smooth(noisy_data.train_data, 5)
    dif = noisy_data.train_data[:, -1, :] - noisy_data.train_data[:, -2, :]
    sum_ = 0
    for pos in range(test_range):
        # print('{} \t'.format(np.sum((noisy_data.train_data[:, -1, :] - noisy_data.test_data[:, pos, :]) ** 2) /
        #                      np.sum(noisy_data.test_data[:, pos, :]) ** 2))
        # pred = (pos + 1) * dif + noisy_data.train_data[:, -1, :]
        # print('{} \t'.format(np.sum((pred - noisy_data.test_data[:, pos, :]) ** 2) /
        #                      np.sum(noisy_data.test_data[:, pos, :]) ** 2))
        # print(np.sum(pred))

        print('{} \t'.format(np.sum((noisy_data.train_data[:, -1, :] - noisy_data.test_data[:, pos, :]) ** 2)))
        # sum_ += np.sum((noisy_data.train_data[:, -1, :] - noisy_data.test_data[:, pos, :]) ** 2)
        pred = (pos + 1) * dif + noisy_data.train_data[:, -1, :]
        print('{} \t'.format(np.sum((pred - noisy_data.test_data[:, pos, :]) ** 2)))
        sum_ += np.sum((pred - noisy_data.test_data[:, pos, :]) ** 2)
    # log.write('\n')
    # log.close()
    print(sum_)
    # if iter_p_ > iter_patience:
    #     break


if __name__ == '__main__':
    main(smooth_gh=0.1, smooth_p=True)
