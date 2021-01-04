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
x_min = config.X_MIN
x_max = config.X_MAX
# x_points = config.X_POINTS
# x_gap = (x_max - x_min) / x_points
t_gap = config.T_GAP
sigma = config.SIGMA
# learning_rate_gh = config.LEARNING_RATE_GH
# learning_rate_p = config.LEARNING_RATE_P
learning_rate_p = 0.1 ** 6
# patience = config.PATIENCE
# recur_win_gh = 13
# recur_win_p = 13
verb = 2
train_range = 5
test_range = 5
sf_range = 7
t_sro = 7
epoch = 10000
patience = 10


def test_one_step(x, g, h, p, t):
    dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
    dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)
    # predict_pxt_euler = np.zeros((test_range, x.shape[0]))
    # print('0', p.shape, g.shape, h.shape, t.shape)
    k1 = np.matmul(g * p, dx) + np.matmul(h * p, dxx)
    print('1', p.shape, k1.shape, t.shape)
    # k1.reshape(-1, 1)
    # t.reshape(1, -1)
    delta_p = np.multiply(k1, t)
    # print(delta_p.shape, t)
    # print(k1[50:60], delta_p[:, 50:60])
    predict_pxt_euler = p + delta_p
    print('2', p.shape, k1.shape, t.shape, predict_pxt_euler.shape)
    # print(p0[:5], delta_p[:5, :5], predict_pxt_euler[sample, :5, :5])
    return predict_pxt_euler


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


def main():
    op_dir = '/home/liuwei/GitHub/Result/Boltz/test'
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    # ~~~~~~~~~~~~~~~
    run_ = 9
    iter_ = 134
    ip_dir = '/home/liuwei/GitHub/Result/Boltz/id{}_p{}_win{}{}_{}'.format(1, 10, 13, 13, run_)
    gg = np.load(ip_dir + '/iter{}_gg_ng.npy'.format(iter_))
    hh = np.load(ip_dir + '/iter{}_hh_ng.npy'.format(iter_))
    # ~~~~~~~~~~~~~~~
    gg = gg.reshape((-1, 1, 1))
    hh = hh.reshape((-1, 1, 1))
    # print(gg, hh)

    # ~~~~~~~~~~~~~~~

    data = np.load('./Pxt/Boltz_id{}_{}_sigma{}.npz'.format('test', 1221732, 0.01))
    # data = np.load('./Pxt/Bessel_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    # data = np.load('./Pxt/OU_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    x = data['x']
    t = data['t']
    true_pxt = data['true_pxt']
    noisy_pxt = data['noisy_pxt']
    true_pxt[true_pxt < 0] = 0
    noisy_pxt[noisy_pxt < 0] = 0
    true_pxt /= np.sum(true_pxt[0, 0, :])
    noisy_pxt /= np.sum(noisy_pxt[0, 0, :])
    print(t.shape, true_pxt.shape, noisy_pxt.shape)

    n_sample, t_points, x_points = noisy_pxt.shape
    # sys.exit()

    # ~~~~~~~~~~~~~~~~
    fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
    p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                              fix_g=gg, fix_h=hh)
    train_p_x = np.ones((1, x_points, 1))
    pre_loss = []
    after_loss = []
    err_bt = np.zeros((20*(t_points - test_range - train_range + 1), test_range))
    err_at = np.zeros((20*(t_points - test_range - train_range + 1), test_range))
    print(err_bt.shape, err_at.shape)
    count = 0
    for sample in range(20):
        for t_points in range(train_range-1, t_points-test_range):

            train_p_y = np.copy(noisy_pxt[sample, t_points-train_range+1: t_points+1, :])
            train_p_t = t[sample, t_points-train_range+1: t_points+1, :] - t[sample, t_points, :]

            pred_t = t[sample, t_points+1: t_points + test_range+1, :] - t[sample, t_points, :]
            test_p_y = np.copy(noisy_pxt[sample, t_points+1: t_points+test_range+1, :])

            # print(train_p_y[-1, 50:60])
            train_p_y = train_p_y.transpose()
            # print(train_p_y[50:60, -1])
            train_p_y = train_p_y.reshape((1, x_points, train_range))
            # ~~~~~~~~~~~ smooth
            train_p_y = padding_by_axis2_smooth(train_p_y, 5)           # smooth is better
            # ~~~~~~~~~~~
            # print(train_p_y[0, 50:60, -1])
            train_p_p = np.copy(train_p_y[:, :, -1]).reshape((-1, 1, 1))
            # train_p_p = np.copy(noisy_pxt[sample, t_points, :]).reshape((-1, 1, 1))
            # print(train_p_p[50:60, 0, 0])
            # print(train_p_y[0, :, -1] - train_p_p[:, 0, 0])
            # sys.exit()
            train_p_t = train_p_t.reshape(1, 1, -1)

            p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p])
            es = callbacks.EarlyStopping(verbose=verb, patience=patience)
            print('3', train_p_x.shape, train_p_t.shape, train_p_y.shape)
            # sys.exit()
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t], train_p_y)
            pre_loss.append(p_loss)
            p_nn_ng.fit([train_p_x, train_p_t], train_p_y,
                        epochs=epoch, verbose=verb, callbacks=[es],
                        validation_data=[[train_p_x, train_p_t], train_p_y])
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t], train_p_y)
            after_loss.append(p_loss)
            trained_p = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            pred_p_before_train = test_one_step(x, gg[:, 0, 0], hh[:, 0, 0], train_p_p[:, 0, 0], pred_t)
            err_bt[count, :] = np.sum((pred_p_before_train - test_p_y)**2, axis=1)
            pred_p_after_train = test_one_step(x, gg[:, 0, 0], hh[:, 0, 0], trained_p, pred_t)
            err_at[count, :] = np.sum((pred_p_after_train - test_p_y) ** 2, axis=1)
            fix_err = np.sum((train_p_p[:, 0, 0] - test_p_y) ** 2)
            print(np.sum(err_bt[count]), np.sum(err_at[count]), fix_err)
            count += 1

            # print(pred_p_after_train.shape)
            plt.figure()
            plt.plot(pred_p_after_train[-1], 'k-', label='after train', linewidth=1)
            plt.plot(test_p_y[-1], 'r', label='trained_p', linewidth=1)
            # plt.plot(one_pred[sample, :, 2], 'r-', label='pred', linewidth=1)
            plt.plot(pred_p_before_train[-1], 'b-', label='input', linewidth=1)
            # plt.plot(P[2, 44, :], 'r-', label='trained', linewidth=1)
            # plt.plot(pre_P[1, -1, :], 'b-', label='p_initial', linewidth=1)
            plt.legend()
            # plt.title('iter {}'.format(i))
            plt.show()

            # print(np.sum((pred_p_before_train - test_p_y)**2, axis=1))
            # print(np.sum((pred_p_after_train - test_p_y) ** 2, axis=1))
            sys.exit()
    log = open(op_dir + '/train.log', 'a')
    log.write('{} iter {} \n'.format(op_dir, iter_))
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
    # print(sum(pre_loss), sum(after_loss))


if __name__ == '__main__':
    main()
