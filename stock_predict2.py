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

# import OU_config as config
# import B_config as config
import Boltz_config as config

from GridModules.GaussianSmooth import GaussianSmooth

# np.set_printoptions(suppress=True)

name = 'Noisy'
seed = config.SEED

x_min = -0.015
x_max = 0.015
x_points = 100
t_points = 30

# x_gap = (x_max - x_min) / x_points
t_gap = 1
# learning_rate_gh = config.LEARNING_RATE_GH
# learning_rate_p = config.LEARNING_RATE_P
learning_rate_p = 0.1 ** 5
# patience = config.PATIENCE
# recur_win_gh = 13
# recur_win_p = 13
verb = 2
train_range = 3
test_range = 5
sf_range = 7
t_sro = 7
epoch = 10000
patience = 10

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

# def test_one_step(x, g, h, p, t):
#     dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
#     dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)
#     # predict_pxt_euler = np.zeros((test_range, x.shape[0]))
#     # print('0', p.shape, g.shape, h.shape, t.shape)
#     k1 = np.matmul(g * p, dx) + np.matmul(h * p, dxx)
#     print('1', p.shape, k1.shape, t.shape)
#     # k1.reshape(-1, 1)
#     # t.reshape(1, -1)
#     delta_p = np.multiply(k1, t)
#     # print(delta_p.shape, t)
#     # print(k1[50:60], delta_p[:, 50:60])
#     predict_pxt_euler = p + delta_p
#     print('2', p.shape, k1.shape, t.shape, predict_pxt_euler.shape)
#     # print(p0[:5], delta_p[:5, :5], predict_pxt_euler[sample, :5, :5])
#     return predict_pxt_euler


def padding_by_axis2_smooth(data, size):
    data_shape = list(data.shape)
    print(data_shape)
    data_shape[1] = int(data_shape[1] + 2 * size)                           # dim from 2 to 1
    data_shape = tuple(data_shape)
    expand = np.zeros(data_shape)
    expand[:, size: -size, :] = data                                        # dim from 2 to 1
    expand = signal.savgol_filter(expand, train_range, 2, axis=2)           # change from sf_range to train_range
    expand = signal.savgol_filter(expand, train_range, 2, axis=1)           # change from sf_range to train_range
    smooth_data = expand[:, size: -size, :]                                 # dim from 2 to 1
    return smooth_data


def main(stock):
    x = np.linspace(x_min, x_max, num=x_points, endpoint=True)

    data = np.loadtxt('./stock/data_x.dat')
    # data *= 100
    if stock == 'FTSE':
        n_sequence = len(F_frag)
        print(n_sequence)
        noisy_pxt = np.zeros((n_sequence, t_points, x_points))
        for s in range(n_sequence):
            start, end = F_frag[s]
            noisy_pxt[s, :, :] = data[start: end, :100]
    elif stock == 'Nikki':
        n_sequence = len(N_frag)
        print(n_sequence)
        noisy_pxt = np.zeros((n_sequence, t_points, x_points))
        for s in range(n_sequence):
            start, end = N_frag[s]
            noisy_pxt[s, :, :] = data[start: end, 200:]
    elif stock == 'DOW':
        n_sequence = len(N_frag)
        print(n_sequence)
        noisy_pxt = np.zeros((n_sequence, t_points, x_points))
        for s in range(n_sequence):
            start, end = N_frag[s]
            noisy_pxt[s, :, :] = data[start: end, 100:200]

    t = np.zeros((n_sequence, t_points, 1))
    for i in range(t_points):
        t[:, i, :] = i * t_gap

    op_dir = '/home/liuwei/GitHub/Result/Stock/test_{}'.format(stock)
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    noisy_pxt = noisy_pxt[22:27, ...]
    t = t[22:27, ...]
    # ~~~~~~~~~~~~~~~

    run_ = 0
    iter_ = 122
    # ip_dir = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_v3'.format(stock, 20, 5, 5, run_)
    ip_dir = '/home/liuwei/Cluster/Stock/{}_p{}_win{}{}_{}_v5'.format(stock, 20, 5, 5, run_)
    # gg = np.load(ip_dir + '/iter{}_gg_ng.npy'.format(iter_))
    # hh = np.load(ip_dir + '/iter{}_hh_ng.npy'.format(iter_))
    # ~~~~~~~~~~~~~~~
    npz_path = ip_dir + '/iter{}.npz'.format(iter_)
    npz = np.load(npz_path)
    gg = npz['g']
    hh = npz['h']
    # stock = npz['P']
    # print(np.min(stock), np.max(stock))
    # sys.exit()
    gg = gg.reshape((-1, 1, 1))
    hh = hh.reshape((-1, 1, 1))
    # print(gg, hh)

    # ~~~~~~~~~~~~~~~

    n_sample, _, _ = noisy_pxt.shape
    # sys.exit()

    # ~~~~~~~~~~~~~~~~
    fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
    p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                              fix_g=gg, fix_h=hh)
    train_p_x = np.ones((1, x_points, 1))
    pre_loss = []
    after_loss = []
    err_bt = np.zeros((n_sample * (t_points - test_range - train_range + 1), test_range))
    err_at = np.zeros((n_sample * (t_points - test_range - train_range + 1), test_range))
    err_fix = np.zeros((n_sample * (t_points - test_range - train_range + 1), test_range))
    print(err_bt.shape, err_at.shape)
    count = 0
    for sample in range(n_sample):
        for t_idx in range(train_range-1, t_points-test_range):
            train_p_y = np.copy(noisy_pxt[sample, t_idx-train_range+1: t_idx+1, :])
            train_p_t = t[sample, t_idx-train_range+1: t_idx+1, :] - t[sample, t_idx, :]

            pred_t = t[sample, t_idx+1: t_idx + test_range+1, :] - t[sample, t_idx, :]
            test_p_y = np.copy(noisy_pxt[sample, t_idx+1: t_idx+test_range+1, :])

            # print(train_p_y[-1, 50:60])
            train_p_y = train_p_y.transpose()
            # print(train_p_y[50:60, -1])
            train_p_y = train_p_y.reshape((1, x_points, train_range))
            # ~~~~~~~~~~~ smooth
            # train_p_y = padding_by_axis2_smooth(train_p_y, 5)           # smooth is better
            # ~~~~~~~~~~~
            # print(train_p_y[0, 50:60, -1])
            train_p_p = np.copy(train_p_y[:, :, -1]).reshape((-1, 1, 1))
            # train_p_p = np.copy(noisy_pxt[sample, t_points, :]).reshape((-1, 1, 1))
            # print(train_p_p[50:60, 0, 0])
            # print(train_p_y[0, :, -1] - train_p_p[:, 0, 0])
            # sys.exit()
            train_p_t = train_p_t.reshape(1, 1, -1)

            p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p])

            print('3', train_p_x.shape, train_p_t.shape, train_p_y.shape)
            # sys.exit()
            # ~~~~~~~~~~~~~~~~ loss before train
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t], train_p_y)
            pre_loss.append(p_loss)

            # ~~~~~~~~~~~~~~~~ prediction before train p
            pred_t = pred_t.reshape(1, 1, -1)
            pred_p_before_train = p_nn_ng.predict([train_p_x, pred_t])
            pred_p_before_train = pred_p_before_train[0, ...].transpose()

            # ~~~~~~~~~~~~~~~~ train
            es = callbacks.EarlyStopping(verbose=verb, patience=patience)
            p_nn_ng.fit([train_p_x, train_p_t], train_p_y,
                        epochs=epoch, verbose=verb, callbacks=[es],
                        validation_data=[[train_p_x, train_p_t], train_p_y])

            # ~~~~~~~~~~~~~~~~ loss after train
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t], train_p_y)
            after_loss.append(p_loss)

            # ~~~~~~~~~~~~~~~~ prediction after train p
            pred_p_after_train = p_nn_ng.predict([train_p_x, pred_t])
            pred_p_after_train = pred_p_after_train[0, ...].transpose()

            # ~~~~~~~~~~~~~~~~ calculate error
            err_bt[count, :] = np.sum((pred_p_before_train - test_p_y)**2, axis=1)
            err_at[count, :] = np.sum((pred_p_after_train - test_p_y) ** 2, axis=1)
            err_fix[count, :] = np.sum((train_p_p[:, 0, 0] - test_p_y) ** 2, axis=1)
            # er = np.sum((train_p_p[:, 0, 0] - test_p_y) ** 2)
            print(np.sum(err_bt[count]), np.sum(err_at[count]), np.sum(err_fix[count]))
            count += 1

            # print(train_p_p[50:60, 0, 0])
            # print(err_fix[count-1, :])
            # sys.exit()
            # pred_t = pred_t.reshape(1, 1, -1)
            # y_model = p_nn_ng.predict([train_p_x, pred_t])
            # y_model = y_model[0, ...].transpose()
            # print(np.sum((pred_p_after_train - y_model) ** 2))
            # print(np.max(y_model), np.max(pred_p_after_train))

            # print(pred_p_after_train[:, 50:60])
            # print(pred_t)
            # plt.figure()
            plt.plot(pred_p_after_train[2], 'r-', label='after train', linewidth=1)
            plt.plot(pred_p_after_train[0], 'b-', label='after train', linewidth=1)
            # plt.plot(pred_p_before_train[-1], 'b-', label='before train', linewidth=1)
            # plt.plot(pred_p_before_train[0], 'b^', label='before train', linewidth=1)
            # plt.plot(test_p_y[-1], 'k-', label='true_p', linewidth=1)
            # plt.plot(test_p_y[0], 'k^', label='true_p', linewidth=1)
            # plt.plot(train_p_p[:, 0, 0], 'g+', label='fix_p', linewidth=1)
            # plt.plot(one_pred[sample, :, 2], 'r-', label='pred', linewidth=1)

            # plt.plot(P[2, 44, :], 'r-', label='trained', linewidth=1)
            # plt.plot(pre_P[1, -1, :], 'b-', label='p_initial', linewidth=1)
            plt.legend()
            # plt.title('iter {}'.format(i))
            plt.show()

            print(np.sum((pred_p_before_train - test_p_y)**2, axis=1))
            print(np.sum((pred_p_after_train - test_p_y) ** 2, axis=1))
            print(np.sum((train_p_p[:, 0, 0] - test_p_y) ** 2, axis=1))
            # sys.exit()
    print(count, 20*(t_points - test_range - train_range + 1))
    print('first 10', err_fix[:10, :])
    print('last 10', err_fix[-10:, :])
    log = open(op_dir + '/train.log', 'a')
    log.write('{} iter {} \n'.format(ip_dir, iter_))
    log.write('lr {} \t epoch {} \t patient {} \t train range {} \t test range {} \n'
              .format(learning_rate_p, epoch, patience, train_range, test_range))
    log.write('p loss before {} after {} \n'.format(np.sum(pre_loss), np.sum(after_loss)))
    log.write('predict error before mean std: 1 {:.3e} {:.3e}, 2 {:.3e} {:.3e}, 3 {:.3e} {:.3e}, 4 {:.3e} {:.3e},'
              ' 5 {:.3e} {:.3e}\n'.format(np.mean(err_bt[:, 0]), np.std(err_bt[:, 0]),
                                          np.mean(err_bt[:, 1]), np.std(err_bt[:, 1]),
                                          np.mean(err_bt[:, 2]), np.std(err_bt[:, 2]),
                                          np.mean(err_bt[:, 3]), np.std(err_bt[:, 3]),
                                          np.mean(err_bt[:, 4]), np.std(err_bt[:, 4])))
    log.write('predict error after mean std: 1 {:.3e} {:.3e}, 2 {:.3e} {:.3e}, 3 {:.3e} {:.3e}, 4 {:.3e} {:.3e},'
              ' 5 {:.3e} {:.3e}\n'.format(np.mean(err_at[:, 0]), np.std(err_at[:, 0]),
                                          np.mean(err_at[:, 1]), np.std(err_at[:, 1]),
                                          np.mean(err_at[:, 2]), np.std(err_at[:, 2]),
                                          np.mean(err_at[:, 3]), np.std(err_at[:, 3]),
                                          np.mean(err_at[:, 4]), np.std(err_at[:, 4])))
    log.write('predict error fix mean std: 1 {:.3e} {:.3e}, 2 {:.3e} {:.3e}, 3 {:.3e} {:.3e}, 4 {:.3e} {:.3e},'
              ' 5 {:.3e} {:.3e}\n'.format(np.mean(err_fix[:, 0]), np.std(err_fix[:, 0]),
                                          np.mean(err_fix[:, 1]), np.std(err_fix[:, 1]),
                                          np.mean(err_fix[:, 2]), np.std(err_fix[:, 2]),
                                          np.mean(err_fix[:, 3]), np.std(err_fix[:, 3]),
                                          np.mean(err_fix[:, 4]), np.std(err_fix[:, 4])))
    # print(sum(pre_loss), sum(after_loss))


if __name__ == '__main__':
    main(stock='FTSE')
