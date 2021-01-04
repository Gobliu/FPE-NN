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
t_gap = 1

learning_rate_gh = 1e-9
gh_epoch = 10000
gh_patience = 20
batch_size = 32
recur_win_gh = 5

learning_rate_p = 1e-3
p_epoch_factor = 5
recur_win_p = 5

# valid_win = 9
verb = 2

n_iter = 5000
iter_patience = 20
test_range = 0
sf_range = 5        # 7
t_sro = 7

t_point = 30

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

D_frag = [[0, 30], [30, 60], [60, 90], [128, 158], [158, 188], [188, 218], [218, 248], [274, 304], [304, 334],
          [334, 364], [409, 439], [477, 507], [533, 563], [563, 593], [593, 623], [723, 753], [769, 799], [799, 829],
          [829, 859], [859, 889], [890, 920], [920, 950], [970, 1000], [1076, 1106], [1106, 1136], [1136, 1166],
          [1166, 1196], [1196, 1226], [1233, 1263], [1281, 1311], [1330, 1360], [1416, 1446], [1446, 1476],
          [1476, 1506], [1524, 1554], [1554, 1584], [1584, 1614], [1627, 1657], [1661, 1691], [1691, 1721],
          [1721, 1751], [1770, 1800], [1800, 1830], [1845, 1875], [1875, 1905], [1916, 1946], [1946, 1976],
          [1976, 2006], [2006, 2036], [2036, 2066], [2066, 2096], [2096, 2126], [2134, 2164], [2164, 2194],
          [2194, 2224], [2224, 2254], [2297, 2327], [2327, 2357]]


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


def main(stock, smooth_gh=0.1, smooth_p=False):
    run_ = 0
    while run_ < 100:
        directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_v5'.format(stock, gh_patience, recur_win_gh,
                                                                                   recur_win_p, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break

    x = np.linspace(x_min, x_max, num=x_points, endpoint=True)

    data = np.loadtxt('./stock/data_x.dat')
    # data *= 100
    if stock == 'FTSE':
        n_sequence = len(F_frag)
        print(n_sequence)
        noisy_pxt = np.zeros((n_sequence, t_point, x_points))
        for s in range(n_sequence):
            start, end = F_frag[s]
            noisy_pxt[s, :, :] = data[start: end, :100]
    elif stock == 'Nikki':
        n_sequence = len(N_frag)
        print(n_sequence)
        noisy_pxt = np.zeros((n_sequence, t_point, x_points))
        for s in range(n_sequence):
            start, end = N_frag[s]
            noisy_pxt[s, :, :] = data[start: end, 200:]
    elif stock == 'DOW':
        n_sequence = len(N_frag)
        print(n_sequence)
        noisy_pxt = np.zeros((n_sequence, t_point, x_points))
        for s in range(n_sequence):
            start, end = N_frag[s]
            noisy_pxt[s, :, :] = data[start: end, 100:200]

    t = np.zeros((n_sequence, t_point, 1))
    for i in range(t_point):
        t[:, i, :] = i * t_gap

    print(np.sum(noisy_pxt[-1, -1, :]))

    # ~~~~~~~~~~~~~~ split by sequence
    noisy_pxt = noisy_pxt[:20, ...]
    t = t[:20, ...]

    log = open(directory + '/train.log', 'w')
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('t_sro: {}\n'.format(t_sro))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, sf_range))
    smooth_pxt = copy.copy(noisy_pxt)
    smooth_pxt[:, :, :-test_range] = padding_by_axis2_smooth(smooth_pxt[:, :, :-test_range], 5)
    log.write('Difference of pxt after smooth: {} ratio {}\n'.
              format(np.sum((smooth_pxt - noisy_pxt)**2)**0.5,
                     np.sum((smooth_pxt - noisy_pxt)**2)**0.5/np.sum(noisy_pxt**2)**0.5))
    log.close()

    if smooth_p:
        update_pxt = np.copy(smooth_pxt)
    else:
        update_pxt = np.copy(noisy_pxt)
    # update_pxt = np.copy(noisy_pxt)

    noisy_data = PxtData_NG(t=t, x=x, data=noisy_pxt)
    smooth_data = PxtData_NG(t=t, x=x, data=smooth_pxt)
    update_data = PxtData_NG(t=t, x=x, data=update_pxt)
    # update_data_valid = PxtData_NG(t=t, x=x, data=update_pxt)

    # end 2 end
    noisy_data.sample_train_split_e2e(test_range=0)
    smooth_data.sample_train_split_e2e(test_range=0)
    update_data.sample_train_split_e2e(test_range=0)

    # print(noisy_data.train_data.shape, noisy_data.test_data.shape)
    # sys.exit()

    lsq = FPLeastSquare_NG(x_coord=x, t_sro=t_sro)

    if smooth_p:
        lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=smooth_data.train_data, t=smooth_data.train_t)
    else:
        lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=noisy_data.train_data, t=noisy_data.train_t)

    # print(noisy_data.train_data.shape, noisy_data.train_t.shape)
    # sys.exit()
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

    # train gh not end2end, train p end2end
    win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_center(update_data.train_data, update_data.train_t, recur_win_gh)
    train_gh_x_ng = np.copy(win_x)
    train_gh_y_ng = np.copy(win_y)
    train_gh_t_ng = np.copy(win_t)

    win_x, win_t, win_y, win_id = PxtData_NG.get_recur_win_center(noisy_data.train_data, noisy_data.train_t,
                                                                  recur_win_p)
    # win_x, win_t, win_y, win_id = PxtData_NG.get_recur_win_e2e_cw(noisy_data.train_data, noisy_data.train_t,
    #                                                               recur_win_p, cw)
    train_p_p_ng = np.copy(win_x)
    train_p_y_ng = np.copy(win_y)
    train_p_t_ng = np.copy(win_t)

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
        # print(valid_gh_y[0, :, 0], y_model[0, :, 0])
        L_gh_test = gh_nn_ng.evaluate([train_gh_x_ng, train_gh_t_ng], train_gh_y_ng)
        print('Shape of y_model:', y_model.shape, train_p_y_ng.shape)

        log.write('L_gh: {} {} {}\n'.format(L_gh, L_gh_test, L_gh/L_gh_test))

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
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                      train_p_y_ng[sample:sample + 1, ...])
            total_train_p_loss_before += p_loss
            p_nn_ng.fit([train_p_x, train_p_t_ng[sample:sample + 1, ...]], train_p_y_ng[sample:sample + 1, ...],
                        epochs=iter_ // p_epoch_factor + 1, verbose=verb, callbacks=[es],
                        validation_data=[[train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                         train_p_y_ng[sample:sample + 1, ...]])

            temp = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            temp[temp < 0] = 0
            update_data.train_data[sample_id, t_id] = temp / np.sum(temp)
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                      train_p_y_ng[sample:sample + 1, ...])
            total_train_p_loss_after += p_loss

        L_P = total_train_p_loss_after

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_center(update_data.train_data, update_data.train_t,
                                                                 recur_win_gh)
        train_gh_x_ng = np.copy(win_x)
        train_gh_y_ng = np.copy(win_y)
        train_gh_t_ng = np.copy(win_t)  # key??

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_center(update_data.train_data, update_data.train_t,
                                                                 recur_win_p)
        # win_x, win_t, win_y, win_id = PxtData_NG.get_recur_win_e2e_cw(noisy_data.train_data, noisy_data.train_t,
        #                                                               recur_win_p, cw)
        train_p_p_ng = np.copy(win_x)

        log = open(directory + '/train.log', 'a')
        log.write('Total error of p training before: {}, after: {}, Dif P {} {}, Total loss: {}\n'
                  .format(total_train_p_loss_before, total_train_p_loss_after,
                          np.sum((update_data.train_data - noisy_data.train_data)**2),
                          np.sum((update_data.train_data[:, -1, :] - noisy_data.train_data[:, -1, :]) ** 2),
                          L_gh + L_P))
        log.close()

        if L_gh + L_P < total_loss:
            # save
            np.savez_compressed(directory + '/iter{}'.format(iter_),
                                g=gg_v_ng[:, 0, 0], h=hh_v_ng[:, 0, 0], P=update_data.train_data)
            iter_p_ = 0
            total_loss = L_gh + L_P
        else:
            iter_p_ += 1

        if iter_p_ > iter_patience:
            break


if __name__ == '__main__':
    main(stock='FTSE', smooth_gh=0.1, smooth_p=True)
    # stock could choose 'FTSE', 'DOW', 'Nikki'
