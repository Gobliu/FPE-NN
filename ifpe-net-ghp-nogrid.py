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
import B_config as config
# import Boltz_config as config

from GridModules.GaussianSmooth import GaussianSmooth

np.set_printoptions(suppress=True)

name = 'Noisy'
seed = config.SEED
# x_min = config.X_MIN
# x_max = config.X_MAX
# x_points = config.X_POINTS
# x_gap = (x_max - x_min) / x_points
t_gap = config.T_GAP
sigma = config.SIGMA
learning_rate_gh = config.LEARNING_RATE_GH
learning_rate_p = config.LEARNING_RATE_P
gh_epoch = config.EPOCH
p_epoch = 1
patience = config.PATIENCE
batch_size = config.BATCH_SIZE
recur_win_gh = 13
recur_win_p = 13
verb = 2
p_epoch_factor = 5
gh = 'lsq'         # check
n_iter = 500
test_range = 0
sf_range = 7
t_sro = 7
x_r = [6, 81]     # Bessel 10
# x_r = [14, 87]      # Boltz 1
# x_r = [19, 101]     # OU 1


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


def main(run_id, p_patience, smooth_gh=0.1, smooth_p=False):
    run_ = 0
    while run_ < 100:
        # directory = '/home/liuwei/GitHub/Result/Boltz/id{}_p{}_win{}{}_{}'.format(run_id, p_patience, recur_win_gh,
        #                                                                           recur_win_p, run_)
        directory = '/home/liuwei/GitHub/Result/Bessel/id{}_p{}_win{}{}_{}'.format(run_id, p_patience, recur_win_gh,
                                                                                   recur_win_p, run_)
        # directory = '/home/liuwei/GitHub/Result/OU/id{}_p{}_win{}{}_{}'.format(run_id, p_patience, recur_win_gh,
        #                                                                        recur_win_p, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break

    # data = np.load('./Pxt/Boltz_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    data = np.load('./Pxt/Bessel_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    # data = np.load('./Pxt/OU_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    x = data['x']
    x_points = x.shape[0]
    print(x)
    # sys.exit()
    t = data['t']
    print(t.shape)
    # print(t[:10])

    true_pxt = data['true_pxt']
    noisy_pxt = data['noisy_pxt']
    print(true_pxt.shape)
    # sys.exit()
    true_pxt[true_pxt < 0] = 0
    noisy_pxt[noisy_pxt < 0] = 0

    log = open(directory + '/train.log', 'w')
    log.write('id{}_{}_sigma{}.npz \n'.format(run_id, seed, sigma))
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('t_sro: {} \n'.format(t_sro))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, sf_range))
    log.write('Initial error of pxt before smooth: {} ratio {}\n'.
              format(np.sum((noisy_pxt - true_pxt)**2)**0.5,
                     np.sum((noisy_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))

    smooth_pxt = padding_by_axis2_smooth(noisy_pxt, 5)
    log.write('Initial error of pxt after smooth: {} ratio {}\n'.
              format(np.sum((smooth_pxt - true_pxt)**2)**0.5,
                     np.sum((smooth_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))
    log.close()

    # Boltz
    # real_g = x - 1
    # real_h = 0.2 * x**2
    # Bessel
    real_g = 1/x - 0.2
    real_h = 0.5 * np.ones(x.shape)
    # OU
    # real_g = 2.86 * x
    # real_h = 0.0013 * np.ones(x.shape)

    if smooth_p:
        update_pxt = np.copy(smooth_pxt)
    else:
        update_pxt = np.copy(noisy_pxt)

    true_data = PxtData_NG(t=t, x=x, data=true_pxt)
    noisy_data = PxtData_NG(t=t, x=x, data=noisy_pxt)
    smooth_data = PxtData_NG(t=t, x=x, data=smooth_pxt)
    update_data = PxtData_NG(t=t, x=x, data=update_pxt)
    update_data_ng = PxtData_NG(t=t, x=x, data=update_pxt)

    # end 2 end
    true_data.sample_train_split_e2e(test_range=test_range)
    noisy_data.sample_train_split_e2e(test_range=test_range)
    smooth_data.sample_train_split_e2e(test_range=test_range)
    update_data.sample_train_split_e2e(test_range=test_range)
    update_data_ng.sample_train_split_e2e(test_range=test_range)

    lsq = FPLeastSquare_NG(x_coord=x, t_sro=t_sro)

    if smooth_p:
        lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=smooth_data.train_data, t=smooth_data.train_t)
    else:
        lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=noisy_data.train_data, t=noisy_data.train_t)

    # t_lsq_g, t_lsq_h, dt, p_mat = lsq.lsq_wo_t(pxt=true_data.train_data, t=true_data.train_t)

    # plt.figure()
    # plt.plot(x, lsq_g, 'r*')
    # plt.plot(x, t_lsq_g, 'b+')
    # plt.plot(x, real_g, 'k')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(x, lsq_h, 'r*')
    # plt.plot(x, t_lsq_h, 'b+')
    # plt.plot(x, real_h, 'k')
    # plt.show()

    if gh == 'real':
        gg_v, hh_v = real_g, real_h
    else:
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

    p_weight = noisy_data.train_data.sum(axis=0).sum(axis=0)
    p_weight /= sum(p_weight)
    np.save(directory + '/p_weight.npy', p_weight)

    # train gh not end2end, train p end2end
    win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_center(smooth_data.train_data, smooth_data.train_t, recur_win_gh)
    train_gh_x_ng = np.copy(win_x)
    train_gh_y_ng = np.copy(win_y)
    train_gh_t_ng = np.copy(win_t)
    # print(win_t.shape, win_x.shape, win_y.shape)
    # print(win_y[0, 0, :])
    # # print(win_t[1])
    # # print(win_t[2])
    # print(win_y[44, 0, :])
    # # print(win_t[45])
    # # print(win_t[46])
    # sys.exit()
    win_x, win_t, win_y, win_id = PxtData_NG.get_recur_win_center(smooth_data.train_data, smooth_data.train_t, recur_win_p)
    train_p_p_ng = np.copy(win_x)
    train_p_y_ng = np.copy(win_y)
    train_p_t_ng = np.copy(win_t)

    log = open(directory + '/train.log', 'a')
    true_train_x, _, _, _ = PxtData_NG.get_recur_win_center(true_data.train_data, true_data.train_t, recur_win_p)
    log.write('Initial error of p: {} \n'.format(np.sum((train_p_p_ng - true_train_x)**2)**0.5))
    log.close()

    n_sample = train_p_p_ng.shape[0]
    for iter_ in range(n_iter):
        log = open(directory + '/train.log', 'a')
        log.write('Iter: {} \n'.format(iter_))

        # smooth
        gg_v_ng[:, 0, 0] = GaussianSmooth.gaussian1d(gg_v_ng[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))
        hh_v_ng[:, 0, 0] = GaussianSmooth.gaussian1d(hh_v_ng[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))
        # gg_v_ng[:, 0, 0] = signal.savgol_filter(gg_v_ng[:, 0, 0], sf_range, 2)
        # hh_v_ng[:, 0, 0] = signal.savgol_filter(hh_v_ng[:, 0, 0], sf_range, 2)

        # train gh
        gh_nn_ng.get_layer(name=name + 'g').set_weights([gg_v_ng])
        gh_nn_ng.get_layer(name=name + 'h').set_weights([hh_v_ng])

        es = callbacks.EarlyStopping(verbose=verb, patience=patience)
        gh_nn_ng.fit([train_gh_x_ng, train_gh_t_ng], train_gh_y_ng, epochs=gh_epoch, batch_size=64, verbose=verb,
                     callbacks=[es], validation_split=0.2)

        gg_v_ng = gh_nn_ng.get_layer(name=name + 'g').get_weights()[0]
        hh_v_ng = gh_nn_ng.get_layer(name=name + 'h').get_weights()[0]

        y_model_ng = gh_nn_ng.predict([train_gh_x_ng, train_gh_t_ng])
        print('Shape of y_model:', y_model_ng.shape, train_p_y_ng.shape)
        print('Error from gh training', np.sum((gg_v_ng[:, 0, 0] - real_g[:])**2),
              np.sum((hh_v_ng[:, 0, 0] - real_h[:])**2),
              np.sum((train_gh_y_ng - y_model_ng)**2))
        # sys.exit()
        log.write('Below are Non Grid.')
        log.write('gh training: {}\n'.format(np.sum((train_gh_y_ng - y_model_ng)**2)))
        log.write('Ratio Error of g: {}, h: {}\n'.format(np.sum((gg_v_ng[:, 0, 0] - real_g)**2)/np.sum(real_g**2),
                                                         np.sum((hh_v_ng[:, 0, 0] - real_h)**2)/np.sum(real_h**2)))
        log.write('Weighted Error of g: {}, h: {}\n'.format(
                                    np.sum(p_weight*(gg_v_ng[:, 0, 0] - real_g)**2)/np.sum(p_weight*real_g**2),
                                    np.sum(p_weight*(hh_v_ng[:, 0, 0] - real_h)**2)/np.sum(p_weight*real_h**2)))
        log.write('Center Error of g: {}, h: {}\n'.format(
                  np.sum((gg_v_ng[x_r[0]:x_r[1], 0, 0] - real_g[x_r[0]:x_r[1]])**2)/np.sum(real_g[x_r[0]:x_r[1]]**2),
                  np.sum((hh_v_ng[x_r[0]:x_r[1], 0, 0] - real_h[x_r[0]:x_r[1]])**2)/np.sum(real_h[x_r[0]:x_r[1]]**2)))
        log.write('Error of g: {}, h: {} \n'.format(np.sum((gg_v_ng[:, 0, 0] - real_g)**2),
                                                    np.sum((hh_v_ng[:, 0, 0] - real_h)**2)))

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
            es = callbacks.EarlyStopping(monitor='loss', verbose=verb, patience=p_patience,
                                         restore_best_weights=True)
            # May 10 change iter_//5
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                      train_p_y_ng[sample:sample + 1, ...])
            total_train_p_loss_before += p_loss
            p_nn_ng.fit([train_p_x, train_p_t_ng[sample:sample + 1, ...]], train_p_y_ng[sample:sample + 1, ...],
                        epochs=iter_ // p_epoch_factor + 1, verbose=verb, callbacks=[es])
            # p_nn_ng.fit([train_p_x, train_p_t_ng[sample:sample + 1, ...]], train_p_y_ng[sample:sample + 1, ...],
            #             epochs=2000, verbose=verb, callbacks=[es],
            #             validation_data=[[train_p_x, train_p_t_ng[sample:sample + 1, ...]],
            #                              train_p_y_ng[sample:sample + 1, ...]])

            update_data_ng.train_data[sample_id, t_id] = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            p_loss = p_nn_ng.evaluate([train_p_x, train_p_t_ng[sample:sample + 1, ...]],
                                      train_p_y_ng[sample:sample + 1, ...])
            total_train_p_loss_after += p_loss

            # y_model_ng = p_nn_ng.predict([train_p_x, train_p_t_ng[sample:sample + 1, ...]])
            # predict_loss += np.sum((y_model_ng - train_p_y_ng[sample:sample + 1, ...]) ** 2)

        # update_data_ng.train_data = padding_by_axis2_smooth(update_data_ng.train_data, 5)

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_center(update_data_ng.train_data, update_data_ng.train_t,
                                                                 recur_win_gh)
        train_gh_x_ng = np.copy(win_x)
        train_gh_y_ng = np.copy(win_y)
        train_gh_t_ng = np.copy(win_t)  # key??

        win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_center(update_data_ng.train_data, update_data_ng.train_t,
                                                                 recur_win_p)
        train_p_p_ng = np.copy(win_x)

        log = open(directory + '/train.log', 'a')
        log.write('Total error of p training before: {}, after: {}, predict: {}\n'.format(total_train_p_loss_before,
                                                                                          total_train_p_loss_after,
                                                                                          predict_loss))
        log.write('Error to true p: {} ratio {}, Error to noisy p: {} ratio {} \n'.
                  format(np.sum((train_gh_x_ng - true_train_x)**2)**0.5,
                         (np.sum((train_gh_x_ng - true_train_x) ** 2)/np.sum(true_train_x**2))**0.5,
                         np.sum((train_p_p_ng - true_train_x) ** 2)**0.5,
                         (np.sum((train_p_p_ng - true_train_x) ** 2)/np.sum(true_train_x**2))**0.5))

        predict_test_euler = test_steps(x, gg_v_ng[:, 0, 0], hh_v_ng[:, 0, 0], update_data_ng)

        log.write('To True data, euler: \t')
        for pos in range(test_range):
            log.write('{} \t'.format(np.sum((predict_test_euler[:, pos, :] - true_data.test_data[:, pos, :]) ** 2)))
        log.write('\n')
        log.write('To Noisy data, euler: \t')
        for pos in range(test_range):
            log.write('{} \t'.format(np.sum((predict_test_euler[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2)))
        log.write('\n')
        log.close()

        # save
        np.save(directory + '/iter{}_gg_ng.npy'.format(iter_), gg_v_ng[:, 0, 0])
        np.save(directory + '/iter{}_hh_ng.npy'.format(iter_), hh_v_ng[:, 0, 0])


if __name__ == '__main__':
    main(run_id=config.RUN_ID, p_patience=10, smooth_gh=0.1, smooth_p=True)
