import sys
import os
import copy

import numpy as np
from scipy import signal
from keras import callbacks, backend, losses
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_List import PxtData_List
from NonGridModules.FPLeastSquare_NG import FPLeastSquare_NG
from NonGridModules.FPENet_NG import FPENet_NG
from NonGridModules.Loss import Loss

from GridModules.GaussianSmooth import GaussianSmooth

np.set_printoptions(suppress=True)

name = 'Noisy'
x_min = -0.015
x_max = 0.015
x_points = 100
t_gap = 0.001

learning_rate_gh = 1e-5
gh_epoch = 10000
gh_patience = 20
batch_size = 32
recur_win_gh = 5

learning_rate_p = 1e-3
p_epoch_factor = 5
recur_win_p = 5

verb = 2

n_iter = 5000
iter_patience = 20
test_range = 5
sf_range = 5        # 7
t_sro = 7

n_sequence = 40
day_no = 2370
cw = 0

FTSE_fragment = [[0, 41], [42, 85], [94, 141], [142, 188], [189, 231], [232, 273], [286, 328], [329, 372],
                 [389, 438], [439, 488], [497, 532], [533, 572], [573, 616], [617, 639], [691, 730],
                 [731, 770], [771, 800], [801, 840], [841, 884], [908, 969], [990, 1023], [1042, 1073],
                 [1074, 1107], [1108, 1141], [1142, 1182], [1184, 1220], [1221, 1259], [1264, 1324],
                 [1402, 1441], [1442, 1481], [1482, 1523], [1524, 1563], [1564, 1600], [1601, 1640],
                 [1641, 1680], [1681, 1720], [1721, 1753], [1770, 1803], [1804, 1840], [1845, 1885],
                 [1886, 1925], [1926, 1965], [1966, 2005], [2006, 2045], [2046, 2083], [2137, 2177],
                 [2178, 2213], [2214, 2247], [2248, 2286], [2313, 2369]]


def test_one_euler(x, g, h, data):
    dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
    dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)

    n_sample = len(data.test_data)
    predict_pxt_euler = np.zeros((n_sample, test_range, x.shape[0]))
    g_truth = np.zeros((n_sample, test_range, x.shape[0]))
    for sample in range(n_sample):
        print(sample, data.test_data[sample].shape)
        g_truth[sample] = np.copy(data.test_data[sample])
        p0 = data.train_data[sample][-1, :]
        relative_t = data.test_t[sample] - data.train_t[sample][-1, :]
        k1 = np.matmul(g * p0, dx) + np.matmul(h * p0, dxx)
        k1.reshape(-1, 1)
        relative_t.reshape(1, -1)
        delta_p = np.multiply(k1, relative_t)
        predict_pxt_euler[sample] = p0 + delta_p
    return predict_pxt_euler, g_truth


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
        directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_v2'.format(stock, gh_patience, recur_win_gh,
                                                                                   recur_win_p, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break

    x = np.linspace(x_min, x_max, num=x_points, endpoint=True)

    data = np.loadtxt('./stock/data_x.dat')
    noisy_pxt = np.zeros((day_no, x_points))

    for i in range(n_sequence):
        if stock == 'FTSE':
            noisy_pxt = data[:, :100]
        elif stock == 'DOW':
            noisy_pxt = data[:, 100:200]
        elif stock == 'Nikki':
            noisy_pxt = data[:, 200:]
        else:
            sys.exit('Stock name is wrong.')

    t = np.zeros((day_no, 1))
    for i in range(day_no):
        t[i, :] = i * t_gap

    log = open(directory + '/train.log', 'w')
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('t_sro: {} \n'.format(t_sro))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, sf_range))
    # log.write('Difference of pxt after smooth: {} ratio {}\n'.
    #           format(np.sum((smooth_pxt - noisy_pxt)**2)**0.5,
    #                  np.sum((smooth_pxt - noisy_pxt)**2)**0.5/np.sum(noisy_pxt**2)**0.5))
    log.close()

    # if smooth_p:
    #     update_pxt = np.copy(smooth_pxt)
    # else:
    #     update_pxt = np.copy(noisy_pxt)
    # update_pxt = np.copy(noisy_pxt)

    noisy_data = PxtData_List(t=t, x=x, data=noisy_pxt, f_list=FTSE_fragment)
    noisy_data.sample_train_split(test_range=test_range)

    update_pxt = np.copy(noisy_pxt)
    update_data = PxtData_List(t=t, x=x, data=update_pxt, f_list=FTSE_fragment)
    update_data.sample_train_split(test_range=test_range)

    lsq = FPLeastSquare_NG(x_coord=x, t_sro=t_sro)

    # print(noisy_data.train_t)
    lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t_list(pxt_list=noisy_data.train_data, t_list=noisy_data.train_t)

    gg_v, hh_v = lsq_g, lsq_h
    # print(gg_v.shape)
    gg_v = np.expand_dims(gg_v, axis=-1)
    gg_v = np.expand_dims(gg_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    # print(gg_v.shape)
    gg_v_ng = np.copy(gg_v)
    hh_v_ng = np.copy(hh_v)

    fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
    gh_nn_ng = fpe_net_ng.recur_train_gh(learning_rate=learning_rate_gh, loss=Loss.sum_square)
    p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                              fix_g=gg_v, fix_h=hh_v)

    train_p_x = np.ones((1, x_points, 1))

    print(noisy_data.train_t[0].shape)

    # train gh not end2end, train p end2end
    win_x, win_t, win_y, _ = update_data.get_recur_win(recur_win_gh)
    # print(win_y[-1, 50:60, :], win_t[-1], win_id[-1], win_x[-1, 50:60])
    train_gh_x_ng = np.copy(win_x)
    train_gh_y_ng = np.copy(win_y)
    train_gh_t_ng = np.copy(win_t)

    win_x, win_t, win_y, win_id = noisy_data.get_recur_win(recur_win_gh)
    # print(win_y[-1, 50:60, :], win_t[-1], win_id[-1], win_x[-1, 50:60])
    train_p_id = np.copy(win_id)
    train_p_p_ng = np.copy(win_x)
    train_p_y_ng = np.copy(win_y)
    train_p_t_ng = np.copy(win_t)
    # sys.exit()

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

        y_model = gh_nn_ng.predict([train_gh_x_ng, train_gh_t_ng])
        L_gh = np.sum((train_gh_y_ng - y_model)**2)
        L_gh_test = gh_nn_ng.evaluate([train_gh_x_ng, train_gh_t_ng], train_gh_y_ng)
        print('Shape of y_model:', y_model.shape, train_p_y_ng.shape)

        log.write('L_gh: {} {} {}\n'.format(L_gh, L_gh_test, L_gh/L_gh_test))

        # train p
        p_nn_ng.get_layer(name=name + 'g').set_weights([gg_v_ng])
        p_nn_ng.get_layer(name=name + 'h').set_weights([hh_v_ng])

        total_train_p_loss_before = 0
        total_train_p_loss_after = 0

        for sample in range(n_sample):
            train_p_id = win_id[sample]  # no true data, end2end
            print('Training P, Sample id: {}'.format(train_p_id))
            p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p_ng[sample].reshape(-1, 1, 1)])
            es = callbacks.EarlyStopping(verbose=verb, patience=gh_patience)
            # May 10 change iter_//5
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                      train_p_y_ng[sample:sample + 1, ...])
            total_train_p_loss_before += p_loss
            p_nn_ng.fit([train_p_x, train_p_t_ng[sample:sample + 1, ...]], train_p_y_ng[sample:sample + 1, ...],
                        epochs=iter_ // p_epoch_factor + 1, verbose=verb, callbacks=[es],
                        validation_data=[[train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                         train_p_y_ng[sample:sample + 1, ...]])

            temp = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            temp[temp < 0] = 0
            update_pxt[train_p_id] = temp / np.sum(temp)
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                      train_p_y_ng[sample:sample + 1, ...])
            total_train_p_loss_after += p_loss

        L_P = total_train_p_loss_after

        update_data = PxtData_List(t=t, x=x, data=update_pxt, f_list=FTSE_fragment)
        update_data.sample_train_split(test_range=test_range)
        win_x, win_t, win_y, _ = update_data.get_recur_win(recur_win_gh)
        train_gh_x_ng = np.copy(win_x)
        train_gh_y_ng = np.copy(win_y)
        train_gh_t_ng = np.copy(win_t)  # key??

        win_x, _, _, _ = update_data.get_recur_win(recur_win_p)
        train_p_p_ng = np.copy(win_x)

        log = open(directory + '/train.log', 'a')
        log.write('Total error of p training before: {}, after: {}, Dif P {}, Total loss: {}\n'
                  .format(total_train_p_loss_before, total_train_p_loss_after,
                          np.sum((noisy_pxt - update_pxt)**2),
                          L_gh + L_P))

        predict_one_euler, ground_t = test_one_euler(x, gg_v_ng[:, 0, 0], hh_v_ng[:, 0, 0], update_data)
        log.write('To Noisy data, one euler: \t')
        for pos in range(test_range):
            log.write('{} \t'.format(np.sum((predict_one_euler[:, pos, :] - ground_t[:, pos, :]) ** 2)))
        log.write('\n')
        log.close()

        if L_gh + L_P < total_loss:
            # save
            np.savez_compressed(directory + '/iter{}'.format(iter_),
                                g=gg_v_ng[:, 0, 0], h=hh_v_ng[:, 0, 0], P=update_data.train_data,
                                predict=predict_one_euler, test=ground_t)
            iter_p_ = 0
            total_loss = L_gh + L_P
        else:
            iter_p_ += 1

        if iter_p_ > iter_patience:
            break


if __name__ == '__main__':
    main(stock='FTSE', smooth_gh=0.1, smooth_p=True)
    # stock could choose 'FTSE', 'DOW', 'Nikki'
