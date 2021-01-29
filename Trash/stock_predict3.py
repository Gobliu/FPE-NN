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
t_gap = 1

learning_rate_gh = 1e-9
gh_epoch = 10000
patience = 20
batch_size = 32
win_gh = 5

learning_rate_p = 1e-5
p_epoch_factor = 5
win_p = 5

# valid_win = 9
verb = 2

n_iter = 5000
iter_patience = 20
t_sro = 7

test_ratio = 0.2
gap_ratio = 0.1
epoch = 1


def test_one_step(x, g, h, p, t):
    dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
    dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)
    # predict_pxt_euler = np.zeros((test_range, x.shape[0]))
    # print('0', p.shape, g.shape, h.shape, t.shape)
    k1 = np.matmul(g * p, dx) + np.matmul(h * p, dxx)
    # print('1', p.shape, k1.shape, t.shape)
    # p = p.reshape(-1, 1)
    # k1 = k1.reshape(-1, 1)
    # t = t.reshape(1, -1)
    # print('2', p.shape, k1.shape, t.shape)
    delta_p = np.multiply(k1, t)
    print(delta_p.shape, t)
    print(k1[50:60], delta_p[:, 50:60])
    predict_pxt_euler = p + delta_p
    # print(p0[:5], delta_p[:5, :5], predict_pxt_euler[sample, :5, :5])
    return predict_pxt_euler


# def test_step_euler(x, g, h, p, test_win):
#     dx = PDM_NG.pde_1d_mat(x, t_sro, sro=1)
#     dxx = PDM_NG.pde_1d_mat(x, t_sro, sro=2)
#     # n_sample = data.test_data.shape[0]
#     # predict_t_points = data.test_data.shape[1]
#     predict_pxt_euler = np.zeros((test_win, x.shape[0]))
#     rep_p = p
#     for t_idx in range(test_win):
#         k1 = np.matmul(g * rep_p, dx) + np.matmul(h * rep_p, dxx)
#         k1.reshape(-1, 1)
#         delta_p = k1 * t_gap
#         rep_p += delta_p
#         rep_p= signal.savgol_filter(rep_p, 5, 2)
#         predict_pxt_euler[t_idx] = np.copy(rep_p)
#     return predict_pxt_euler


def data_spit(stock_type, start, end):
    data_ = np.loadtxt('./stock/data_x.dat')
    if stock_type == 'FTSE':
        stock = data_[start:end, :100]
    elif stock_type == 'Nikki':
        stock = data_[start:end, 200:]
    elif stock_type == 'DOW':
        stock = data_[start:end, 100:200]
    else:
        sys.exit('Unknown stock type.')
    gap_range = int((end - start) * gap_ratio)
    test_range = int((end - start) * test_ratio)

    stock_dif = np.zeros((end - start))
    stock_dif[1:] = np.sum((stock[:-1] - stock[1:]) ** 2, axis=1)       # should start from 0
    print(max(stock_dif), min(stock_dif), np.std(stock_dif), np.mean(stock_dif))        # load mean and std
    dif_mean, dif_std = np.mean(stock_dif), np.std(stock_dif)
    threshold = dif_mean + 2. * dif_std
    stock_dif[stock_dif < threshold] = 0
    print(stock_dif)
    train_stock = np.copy(stock[:-(gap_range+test_range)])
    train_dif = np.copy(stock_dif[:-(gap_range+test_range)])
    test_stock = np.copy(stock[-test_range:])
    test_dif = np.copy(stock_dif[-test_range:])
    print('split data shape', test_stock.shape, train_dif.shape, test_stock.shape, test_dif.shape)
    return train_stock, train_dif, test_stock, test_dif


def padding_by_axis2_smooth(data, size, train_range):
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


def train_process(stock, dif, half_win):
    flag = False
    for idx in range(half_win, stock.shape[0]-half_win):
        print(idx, idx-half_win, idx+half_win+1)
        if np.sum(dif[idx:idx+2*half_win+1]) > 0:
            continue
        else:
            if not flag:
                train_x = stock[idx].reshape(1, 100, 1)
                train_y = stock[idx-half_win: idx+half_win+1, :].transpose().reshape((1, 100, 2*half_win+1))
                train_t = np.arange(-half_win, half_win+1).reshape((1, 1, 2*half_win+1))
                train_id = np.array(idx)
                flag = True
            else:
                new_x = stock[idx].reshape(1, 100, 1)
                new_y = stock[idx-half_win: idx+half_win+1, :].transpose().reshape((1, 100, 2*half_win+1))
                new_t = np.arange(-half_win, half_win+1).reshape((1, 1, 2*half_win+1))
                new_id = np.array(idx)

                train_x = np.vstack((train_x, new_x))
                train_y = np.vstack((train_y, new_y))
                train_t = np.vstack((train_t, new_t))
                train_id = np.vstack((train_id, new_id))
    print('train data shape:', train_x.shape, train_y.shape, train_t.shape, train_id.shape)
    return train_x, train_t, train_y, train_id


def test_process(stock, dif, train_win, test_win):
    train_p = []
    train_y = []
    train_t = []
    test_y = []
    test_t = []
    for idx in range(train_win-1, stock.shape[0]-test_win-1):
        print(idx, idx-train_win+1, idx+test_win+1)
        if np.sum(dif[idx-train_win+1:idx+test_win+1]) > 0:
            continue
        else:
            train_p.append(stock[idx].reshape(1, 100, 1))
            train_y.append(stock[idx-train_win+1: idx+1, :].transpose().reshape((1, 100, train_win)))
            train_t.append(np.arange(-train_win+1, 1).reshape((1, 1, train_win)))
            test_y.append(stock[idx+1: idx+test_win+1, :].reshape((1, test_win, 100))) # no transpose, unlike train_y
            test_t.append(np.arange(1, test_win+1).reshape((1, 1, test_win)))
    # print(train_t[0].shape)
    train_p = np.concatenate(train_p, 0)
    train_y = np.concatenate(train_y, 0)
    train_t = np.concatenate(train_t, 0)
    test_y = np.concatenate(test_y, 0)
    test_t = np.concatenate(test_t, 0)
    # print(train_t.shape)

    print('test data shape:', train_p.shape, train_y.shape, train_t.shape, test_y.shape)
    return train_p, train_t, train_y, test_y, test_t


def main(stock_type, start, end, run_, train_win, test_win):
    op_dir = '/home/liuwei/GitHub/Result/Boltz/test'
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    # ~~~~~~~~~~~~~~~
    x = np.linspace(x_min, x_max, num=x_points, endpoint=True)
    train_p_x = np.ones((1, x_points, 1))
    # ip_dir = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_v4'.format(stock_type, patience, win_gh, win_p, run_)
    ip_dir = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_v3'.format(stock_type, patience, win_gh, win_p, run_)
    train_stock, train_dif, test_stock, test_dif = data_spit(stock_type, start, end)
    train_p, train_t, train_y, test_y, test_t = test_process(test_stock, test_dif, train_win=train_win,
                                                             test_win=test_win)

    # train_p, train_t, train_y, train_id = train_process(train_stock, train_dif, 2)

    fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)

    # for iter_ in range(291, 292):
    for iter_ in [225]:
        npz_path = ip_dir + '/iter{}.npz'.format(iter_)
        if not os.path.exists(npz_path):
            print('{} does not exist.'.format(npz_path))
            continue
        else:
            npz = np.load(npz_path)
            gg = npz['g']
            hh = npz['h']
            # stock = npz['P']
            # print(np.min(stock), np.max(stock))

            # g h smooth
            func = np.poly1d(np.polyfit(x[20:80], gg[20:80], 2))
            fitted_g = func(x)

            func = np.poly1d(np.polyfit(x[20:80], hh[20:80], 2))
            fitted_h = func(x)

            # plt.figure()
            # plt.plot(gg, 'k-', label='gg', linewidth=1)
            # plt.plot(fitted_g, 'r-', label='gg', linewidth=1)
            # plt.show()
            #
            # plt.figure()
            # plt.plot(hh, 'k-', label='gg', linewidth=1)
            # plt.plot(fitted_h, 'r-', label='gg', linewidth=1)
            # plt.show()

            print(np.sum(abs(gg)), np.sum(abs(hh)))

            gg = gg.reshape((-1, 1, 1))
            hh = hh.reshape((-1, 1, 1))
            # gg = fitted_g.reshape((-1, 1, 1))
            # hh = fitted_h.reshape((-1, 1, 1))

            fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
            p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                                      fix_g=gg, fix_h=hh)

            pre_loss = []
            after_loss = []
            err_bt = []
            err_at = []
            err_fix = []

            n_sample = train_p.shape[0]
            for sample in range(n_sample):
                train_p_y = np.copy(train_y[sample:sample+1])
                train_p_t = np.copy(train_t[sample:sample+1])

                # train_p_p = np.copy(train_p[sample]).reshape((-1, 1, 1))

                test_p_y = np.copy(test_y[sample:sample+1]).reshape((-1, 100))
                test_p_t = np.copy(test_t[sample:sample+1])             # 2d [n, 1]

                print(train_p_t.shape, test_p_t.shape)
                # sys.exit()
                # print(test_p_y.shape)
                # sys.exit()
                # print(train_p_y.shape, train_p_t.shape, test_p_y.shape, test_p_t.shape)

                # ~~~~~~~~~~~ smooth
                # train_p_y = padding_by_axis2_smooth(train_p_y, 5, train_range=train_win)           # smooth is better
                # ~~~~~~~~~~~
                train_p_p = np.copy(train_p_y[:, :, -1]).reshape((-1, 1, 1))
                p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p])
                es = callbacks.EarlyStopping(verbose=verb, patience=patience)
                p_loss = p_nn_ng.evaluate([train_p_x, train_p_t], train_p_y)
                pre_loss.append(p_loss)
                p_nn_ng.fit([train_p_x, train_p_t], train_p_y,
                            epochs=epoch, verbose=verb, callbacks=[es],
                            validation_data=[[train_p_x, train_p_t], train_p_y])
                p_loss = p_nn_ng.evaluate([train_p_x, train_p_t], train_p_y)
                after_loss.append(p_loss)
                trained_p = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
                # print(trained_p.shape)
                # y_model = p_nn_ng.predict([train_p_x, train_p_t])
                # print(y_model.shape)
                # sys.exit()
                # pre_p_after_train = p_nn_ng.predict()

                # plt.figure()
                # plt.plot(y_model[0, :, 0], label='after train 0', linewidth=1)
                # plt.plot(y_model[0, :, 1], label='after train 1', linewidth=1)
                # plt.plot(y_model[0, :, 2], label='after train 2', linewidth=1)
                # plt.plot(y_model[0, :, 3], label='after train 3', linewidth=1)
                # plt.plot(y_model[0, :, 4], label='after train 4', linewidth=1)
                # plt.plot(trained_p, label='trained_p', linewidth=1)
                # # plt.plot(one_pred[sample, :, 2], 'r-', label='pred', linewidth=1)
                # plt.plot(train_p_p[:, 0, 0], label='untrained_p', linewidth=1)
                # # plt.plot(P[2, 44, :], 'r-', label='trained', linewidth=1)
                # # plt.plot(pre_P[1, -1, :], 'b-', label='p_initial', linewidth=1)
                # plt.legend()
                # # plt.title('iter {}'.format(i))
                # plt.show()

                # trained_p = signal.savgol_filter(trained_p, 7, 2)
                y_model = p_nn_ng.predict([train_p_x, test_p_t])[0].transpose()

                pred_p_before_train = test_one_step(x, gg[:, 0, 0], hh[:, 0, 0], train_p_p[:, 0, 0],
                                                    test_p_t.reshape((-1, 1)))
                # print(y_model.shape, pred_p_before_train.shape)
                # sys.exit()
                err_bt.append(np.sum((pred_p_before_train - test_p_y)**2, axis=1))
                pred_p_after_train = test_one_step(x, gg[:, 0, 0], hh[:, 0, 0], trained_p, test_p_t.reshape((-1, 1)))
                # smooth
                pred_p_after_train[pred_p_after_train < 0] = 0
                pred_p_after_train = signal.savgol_filter(pred_p_after_train, 7, 2, axis=1)
                # pred_p_after_train = signal.savgol_filter(expand, train_range, 2, axis=1)

                # pred_p_after_train = test_step_euler(x, gg[:, 0, 0], hh[:, 0, 0], trained_p, test_win)
                err_at.append(np.sum((pred_p_after_train - test_p_y) ** 2, axis=1))
                err_fix.append(np.sum((train_p_p[:, 0, 0] - test_p_y) ** 2, axis=1))
                print(np.sum(err_bt[sample]), np.sum(err_at[sample]), np.sum(err_fix[sample]))
                print(np.sum((pred_p_before_train - test_p_y)**2, axis=1))
                print(np.sum((pred_p_after_train - test_p_y) ** 2, axis=1))
                print(np.sum((train_p_p[:, 0, 0] - test_p_y) ** 2, axis=1))

                plt.figure()
                # plt.plot(pred_p_after_train[0], label='after train 0', linewidth=1)
                # plt.plot(pred_p_after_train[1], label='after train 1', linewidth=1)
                plt.plot(pred_p_after_train[2], label='after train 2', linewidth=1)
                # plt.plot(pred_p_after_train[3], label='after train 3', linewidth=1)
                # plt.plot(pred_p_after_train[4], label='after train 4', linewidth=1)
                plt.plot(y_model[0], 'r-', label='y model 2', linewidth=1)
                # plt.plot(y_model[])
                plt.plot(trained_p, label='trained_p', linewidth=1)
                # plt.plot(test_p_y[0], label='test_p 0', linewidth=1)
                # plt.plot(test_p_y[4], label='test_p 4', linewidth=1)
                # plt.plot(one_pred[sample, :, 2], 'r-', label='pred', linewidth=1)
                # plt.plot(train_p_p[:, 0, 0], label='untrained_p', linewidth=1)
                # plt.plot(P[2, 44, :], 'r-', label='trained', linewidth=1)
                # plt.plot(pre_P[1, -1, :], 'b-', label='p_initial', linewidth=1)
                plt.legend()
                # plt.title('iter {}'.format(i))
                plt.show()


                # sys.exit()

        log = open(op_dir + '/train.log', 'a')
        log.write('{} iter {} \n'.format(op_dir, iter_))
        log.write('lr {} \t epoch {} \t patient {} \t train range {} \t test range {} \n'
                  .format(learning_rate_p, epoch, patience, train_win, test_win))
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
    main(stock_type='FTSE', start=0, end=600, run_=2, train_win=3, test_win=5)
    # stock could choose 'FTSE', 'DOW', 'Nikki'

