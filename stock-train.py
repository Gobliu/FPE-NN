import sys
import os
import copy

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

learning_rate_gh = 1e-6
gh_epoch = 1000
gh_patience = 20
batch_size = 32
recur_win_gh = 9

learning_rate_p = 1e-6
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


def main(stock, smooth_gh=0.1, smooth_p=False):
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
        if stock == 'FTSE':
            noisy_pxt[i, :, :] = data[i * t_point: (i+1) * t_point, :100]
        elif stock == 'DOW':
            noisy_pxt[i, :, :] = data[i * t_point: (i + 1) * t_point, :100]
        elif stock == 'Nikki':
            noisy_pxt[i, :, :] = data[i * t_point: (i + 1) * t_point, :100]
        else:
            sys.exit('Stock name is wrong.')
    t = np.zeros((n_sequence, t_point, 1))
    for i in range(t_point):
        t[:, i, :] = i * t_gap

    print(np.sum(noisy_pxt[-1, -1, :]))

    log = open(directory + '/train.log', 'w')
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('t_sro: {} \n'.format(t_sro))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, sf_range))
    smooth_pxt = copy.copy(noisy_pxt)
    smooth_pxt[:, :, :-test_range] = padding_by_axis2_smooth(smooth_pxt[:, :, :-test_range], 5)
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

    lsq = FPLeastSquare_NG(x_coord=x, t_sro=t_sro)

    if smooth_p:
        lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=smooth_data.train_data, t=smooth_data.train_t)
    else:
        lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=noisy_data.train_data, t=noisy_data.train_t)

    gg_v, hh_v = lsq_g, lsq_h

    gg_v = np.expand_dims(gg_v, axis=-1)
    gg_v = np.expand_dims(gg_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    gg_v_ng = np.copy(gg_v)
    hh_v_ng = np.copy(hh_v)

    fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
    gh_nn_ng = fpe_net_ng.recur_train_gh(learning_rate=learning_rate_gh, loss=Loss.sum_square)
    p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                              fix_g=gg_v, fix_h=hh_v)

    train_p_x = np.ones((1, x_points, 1))

    # train gh not end2end, train p end2end
    win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(smooth_data.train_data, smooth_data.train_t, recur_win_gh)
    train_gh_x_ng = np.copy(win_x)
    train_gh_y_ng = np.copy(win_y)
    train_gh_t_ng = np.copy(win_t)

    win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(smooth_data.train_data, smooth_data.train_t, valid_win)
    valid_gh_x = np.copy(win_x)
    valid_gh_y = np.copy(win_y)
    valid_gh_t = np.copy(win_t)

    win_x, win_t, win_y, win_id = PxtData_NG.get_recur_win_e2e(smooth_data.train_data, smooth_data.train_t, recur_win_p)
    train_p_p_ng = np.copy(win_x)
    train_p_y_ng = np.copy(win_y)
    train_p_t_ng = np.copy(win_t)

    win_x, win_t, win_y, win_id = PxtData_NG.get_recur_win_e2e(smooth_data.train_data, smooth_data.train_t, valid_win)
    # valid_p_p = np.copy(win_x)
    valid_p_y = np.copy(win_y)
    valid_p_t = np.copy(win_t)

    n_sample = train_p_p_ng.shape[0]
    iter_p_ = 0
    total_loss = sys.maxsize
    for iter_ in range(n_iter):
        log = open(directory + '/train.log', 'a')
        log.write('Iter: {} \n'.format(iter_))

        # smooth
        gg_v_ng[:, 0, 0] = GaussianSmooth.gaussian1d(gg_v_ng[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))
        hh_v_ng[:, 0, 0] = GaussianSmooth.gaussian1d(hh_v_ng[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))

        # train gh
        gh_nn_ng.get_layer(name=name + 'g').set_weights([gg_v_ng])
        gh_nn_ng.get_layer(name=name + 'h').set_weights([hh_v_ng])

        es = callbacks.EarlyStopping(verbose=verb, patience=gh_patience)
        gh_nn_ng.fit([train_gh_x_ng, train_gh_t_ng], train_gh_y_ng, epochs=gh_epoch, batch_size=batch_size,
                     verbose=verb, callbacks=[es], validation_split=0.2)

        gg_v_ng = gh_nn_ng.get_layer(name=name + 'g').get_weights()[0]
        hh_v_ng = gh_nn_ng.get_layer(name=name + 'h').get_weights()[0]

        y_model = gh_nn_ng.predict([valid_gh_x, valid_gh_t])
        L_gh = np.sum((valid_gh_y - y_model)**2)
        print('Shape of y_model:', y_model.shape, train_p_y_ng.shape)

        log.write('L_gh: {}\n'.format(L_gh))

        # train p
        p_nn_ng.get_layer(name=name + 'g').set_weights([gg_v_ng])
        p_nn_ng.get_layer(name=name + 'h').set_weights([hh_v_ng])
        # backend.set_value(p_nn_ng.optimizer.lr, dyn_learning_rate_p)

        total_train_p_loss_before = 0
        total_train_p_loss_after = 0
        predict_loss = 0
        for sample in range(n_sample):
            sample_id, t_id = win_id[sample]  # no true data, end2end
            print('Training P, Sample id: {}, time id {}'.format(sample_id, t_id))
            p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p_ng[sample].reshape(-1, 1, 1)])
            es = callbacks.EarlyStopping(verbose=verb, patience=gh_patience)
            # May 10 change iter_//5
            p_loss = p_nn_ng.evaluate([train_p_x, valid_p_t[sample:sample + 1, ...]],
                                      valid_p_y[sample:sample + 1, ...])
            total_train_p_loss_before += p_loss
            p_nn_ng.fit([train_p_x, train_p_t_ng[sample:sample + 1, ...]], train_p_y_ng[sample:sample + 1, ...],
                        epochs=iter_ // p_epoch_factor + 1, verbose=verb, callbacks=[es],
                        validation_data=[[train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                         train_p_y_ng[sample:sample + 1, ...]])

            update_data.train_data[sample_id, t_id] = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            p_loss = p_nn_ng.evaluate([train_p_x, valid_p_t[sample:sample + 1, ...]],
                                      valid_p_y[sample:sample + 1, ...])
            total_train_p_loss_after += p_loss

        L_P = total_train_p_loss_after

        # update_data_ng.train_data = padding_by_axis2_smooth(update_data_ng.train_data, 5)

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(update_data.train_data, update_data.train_t,
                                                              recur_win_gh)
        train_gh_x_ng = np.copy(win_x)
        train_gh_y_ng = np.copy(win_y)
        train_gh_t_ng = np.copy(win_t)  # key??

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(update_data.train_data, update_data.train_t,
                                                              recur_win_p)
        train_p_p_ng = np.copy(win_x)

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(update_data.train_data, update_data.train_t,
                                                              valid_win)
        valid_gh_x = np.copy(win_x)
        valid_gh_y = np.copy(win_y)
        valid_gh_t = np.copy(win_t)  # key??

        log = open(directory + '/train.log', 'a')
        log.write('Total error of p training before: {}, after: {}, Total loss: {}\n'.format(total_train_p_loss_before,
                                                                                           total_train_p_loss_after,
                                                                                           L_gh + L_P))

        if L_gh + L_P < total_loss:
            # save
            np.savez_compressed(directory + '/iter{}'.format(iter_),
                                g=gg_v_ng[:, 0, 0], h=hh_v_ng[:, 0, 0], P=update_data.train_data)
            iter_p_ = 0
            total_loss = L_gh + L_P
        else:
            iter_p_ += 1

        predict_one_euler = test_one_euler(x, gg_v_ng[:, 0, 0], hh_v_ng[:, 0, 0], update_data)
        log.write('To Noisy data, one euler: \t')
        for pos in range(test_range):
            log.write('{} \t'.format(np.sum((predict_one_euler[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2)))
        log.write('\n')
        log.close()

        if iter_p_ > iter_patience:
            break


if __name__ == '__main__':
    main(stock='Nikki', smooth_gh=0.1, smooth_p=True,)
    # stock could choose 'FTSE', 'DOW', 'Nikki'
