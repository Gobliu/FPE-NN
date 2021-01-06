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

learning_rate_p = 1e-6
p_epoch_factor = 5
recur_win_p = 5

# valid_win = 9
verb = 2

n_iter = 5000
iter_patience = 20
t_sro = 7

test_ratio = 0.2
gap_ratio = 0.1


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
    threshold = dif_mean + 2 * dif_std
    stock_dif[stock_dif < threshold] = 0
    print(stock_dif)
    train_stock = np.copy(stock[:-(gap_range+test_range)])
    train_dif = np.copy(stock_dif[:-(gap_range+test_range)])
    test_stock = np.copy(stock[-test_range:])
    test_dif = np.copy(stock_dif[-test_range:])
    print('split data shape', test_stock.shape, train_dif.shape, test_stock.shape, test_dif.shape)
    return train_stock, train_dif, test_stock, test_dif


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


def main(stock_type, start, end, smooth_gh=0.1):
    run_ = 0
    while run_ < 100:
        directory = '/home/liuwei/GitHub/Result/Stock/{}_p{}_win{}{}_{}_v4'.format(stock_type, gh_patience,
                                                                                   recur_win_gh, recur_win_p, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break

    x = np.linspace(x_min, x_max, num=x_points, endpoint=True)

    # data = np.loadtxt('./stock/data_x.dat')
    train_stock, train_dif, test_stock, test_dif = data_spit(stock_type, start, end)
    update_train_stock = np.copy(train_stock)

    # ~~~~~~~~~~~~~ for lsq
    train_stock_lsq = train_stock.reshape((1, -1, 100))
    train_t_lsq = np.arange(train_stock.shape[0]).reshape((1, -1, 1))
    lsq = FPLeastSquare_NG(x_coord=x, t_sro=t_sro)
    lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=train_stock_lsq, t=train_t_lsq)

    gg_v, hh_v = lsq_g, lsq_h
    gg_v = np.expand_dims(gg_v, axis=-1)
    gg_v = np.expand_dims(gg_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    # ~~~~~~~~~~~~~

    log = open(directory + '/train.log', 'w')
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('t_sro: {} start: {} end: {}\n'.format(t_sro, start, end))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, 'NA'))
    log.write('Difference of pxt after smooth: {} ratio {}\n'.
              format('NA', 'NA'))
    log.close()

    fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
    gh_nn_ng = fpe_net_ng.recur_train_gh(learning_rate=learning_rate_gh, loss=Loss.sum_square)
    p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                              fix_g=gg_v, fix_h=hh_v)

    train_p_x = np.ones((1, x_points, 1))

    win_x, win_t, win_y, _ = train_process(train_stock, train_dif, recur_win_gh//2)
    train_gh_x = np.copy(win_x)
    train_gh_y = np.copy(win_y)
    train_gh_t = np.copy(win_t)

    # train gh not end2end, train p end2end
    win_x, win_t, win_y, win_id = train_process(train_stock, train_dif, recur_win_gh//2)
    train_p_p = np.copy(win_x)
    train_p_y = np.copy(win_y)
    train_p_t = np.copy(win_t)

    # sys.exit()
    n_sample = train_p_p.shape[0]
    iter_p_ = 0
    total_loss = sys.maxsize
    for iter_ in range(n_iter):
        log = open(directory + '/train.log', 'a')
        log.write('Iter: {} \n'.format(iter_))

        # smooth
        gg_v[:, 0, 0] = GaussianSmooth.gaussian1d(gg_v[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))
        hh_v[:, 0, 0] = GaussianSmooth.gaussian1d(hh_v[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))

        # train gh
        gh_nn_ng.get_layer(name=name + 'g').set_weights([gg_v])
        gh_nn_ng.get_layer(name=name + 'h').set_weights([hh_v])

        es = callbacks.EarlyStopping(verbose=verb, patience=gh_patience)
        gh_nn_ng.fit([train_gh_x, train_gh_t], train_gh_y, epochs=gh_epoch, batch_size=batch_size,
                     verbose=verb, callbacks=[es], validation_split=0.2)

        gg_v_ng = gh_nn_ng.get_layer(name=name + 'g').get_weights()[0]
        hh_v_ng = gh_nn_ng.get_layer(name=name + 'h').get_weights()[0]

        y_model = gh_nn_ng.predict([train_gh_x, train_gh_t])
        L_gh = np.sum((train_gh_y - y_model)**2)
        # print(valid_gh_y[0, :, 0], y_model[0, :, 0])
        L_gh_test = gh_nn_ng.evaluate([train_gh_x, train_gh_t], train_gh_y)
        print('Shape of y_model:', y_model.shape, train_p_y.shape)

        log.write('L_gh: {} {} {}\n'.format(L_gh, L_gh_test, L_gh/L_gh_test))

        # train p
        p_nn_ng.get_layer(name=name + 'g').set_weights([gg_v_ng])
        p_nn_ng.get_layer(name=name + 'h').set_weights([hh_v_ng])
        # backend.set_value(p_nn_ng.optimizer.lr, dyn_learning_rate_p)

        total_train_p_loss_before = 0
        total_train_p_loss_after = 0
        predict_loss = 0
        for sample in range(n_sample):
            # sample_id, t_id = win_id[sample]  # no true data, end2end
            sample_id = win_id[sample, 0]
            print('Training P, Sample id: {}, time id {}'.format(sample_id, 'NA'))
            p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p[sample].reshape(-1, 1, 1)])
            es = callbacks.EarlyStopping(verbose=verb, patience=gh_patience)
            # May 10 change iter_//5
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t[sample:sample + 1, ...]],
                                      train_p_y[sample:sample + 1, ...])
            total_train_p_loss_before += p_loss
            p_nn_ng.fit([train_p_x, train_p_t[sample:sample + 1, ...]], train_p_y[sample:sample + 1, ...],
                        epochs=iter_ // p_epoch_factor + 1, verbose=verb, callbacks=[es],
                        validation_data=[[train_p_x, train_p_t[sample:sample + 1, ...]],
                                         train_p_y[sample:sample + 1, ...]])

            temp = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            temp[temp < 0] = 0
            update_train_stock[sample_id] = temp / np.sum(temp)
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t[sample:sample + 1, ...]],
                                      train_p_y[sample:sample + 1, ...])
            total_train_p_loss_after += p_loss

        L_P = total_train_p_loss_after

        win_x, win_t, win_y, _ = train_process(update_train_stock, train_dif, recur_win_gh // 2)
        train_gh_x = np.copy(win_x)
        train_gh_y = np.copy(win_y)
        train_gh_t = np.copy(win_t)

        win_x, win_t, win_y, _ = train_process(update_train_stock, train_dif, recur_win_gh // 2)
        train_p_p = np.copy(win_x)

        log = open(directory + '/train.log', 'a')
        log.write('Total error of p training before: {}, after: {}, Dif P {}, Total loss: {}\n'
                  .format(total_train_p_loss_before, total_train_p_loss_after,
                          np.sum((train_stock - update_train_stock)**2),
                          L_gh + L_P))

        if L_gh + L_P < total_loss:
            # save
            np.savez_compressed(directory + '/iter{}'.format(iter_),
                                g=gg_v_ng[:, 0, 0], h=hh_v_ng[:, 0, 0], P=update_train_stock)
            iter_p_ = 0
            total_loss = L_gh + L_P
        else:
            iter_p_ += 1

        if iter_p_ > iter_patience:
            break


if __name__ == '__main__':
    main(stock_type='FTSE', start=0, end=2370, smooth_gh=0.1)
    # stock could choose 'FTSE', 'DOW', 'Nikki'
