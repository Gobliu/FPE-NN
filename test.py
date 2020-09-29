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
# from PxtData import PxtData
# from Loss import Loss
# from FPLeastSquare import FPLeastSquare
# from FPNet_ReCur import FPNetReCur
# from GaussianSmooth import GaussianSmooth
# from FokkerPlankEqn import FokkerPlankForward as FPF

# import OU_config as config  # B_config or OU_config
# import B_config as config
import Boltz_config as config
from ifpeModules.PxtData import PxtData
from ifpeModules.PartialDerivativeGrid import PartialDerivativeGrid as PDeGrid
from ifpeModules.FPLeastSquare import FPLeastSquare
from ifpeModules.FPNet_ReCur import FPNetReCur

np.set_printoptions(suppress=True)

name = 'Noisy'
seed = config.SEED
x_min = config.X_MIN
x_max = config.X_MAX
# x_points = config.X_POINTS
# x_gap = (x_max - x_min) / x_points
t_gap = config.T_GAP
sigma = config.SIGMA
learning_rate_gh = config.LEARNING_RATE_GH
learning_rate_p = config.LEARNING_RATE_P
gh_epoch = config.EPOCH
# gh_epoch = 10
p_epoch = 1
patience = config.PATIENCE
batch_size = config.BATCH_SIZE
recur_win_gh = 13
recur_win_p = 13
verb = 2
p_epoch_factor = 5
gh = 'real'  # check
n_iter = 1
test_range = 5
sf_range = 7
t_sro = 7


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
        # directory = './Result/OU/{}_id{}_p{}_win{}{}'.format(run_, run_id, p_patience, recur_win_gh, recur_win_p)
        # directory = './Result/Bessel/id{}_{}_p{}_win{}{}'.format(run_id, run_, p_patience, recur_win_gh, recur_win_p)
        directory = './Result/test/id{}_p{}_win{}{}_{}'.format(run_id, p_patience, recur_win_gh, recur_win_p, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break

    data = np.load('./Pxt/Boltz_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    x = data['x']
    x_points = x.shape[0]
    print(x_points)
    t = data['t']
    true_pxt = data['true_pxt']
    noisy_pxt = data['noisy_pxt']

    true_pxt[true_pxt < 0] = 0
    noisy_pxt[noisy_pxt < 0] = 0

    log = open(directory + '/train.log', 'w')
    log.write('./Pxt/Boltz_id{}_{}_sigma{}.npz \n'.format(run_id, seed, sigma))
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('t_sro: {} \n'.format(t_sro))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, sf_range))
    log.write('Initial error of pxt before smooth: {} ratio {}\n'.
              format(np.sum((noisy_pxt - true_pxt) ** 2) ** 0.5,
                     np.sum((noisy_pxt - true_pxt) ** 2) ** 0.5 / np.sum(true_pxt ** 2) ** 0.5))

    smooth_pxt = padding_by_axis2_smooth(noisy_pxt, 5)
    log.write('Initial error of pxt after smooth: {} ratio {}\n'.
              format(np.sum((smooth_pxt - true_pxt) ** 2) ** 0.5,
                     np.sum((smooth_pxt - true_pxt) ** 2) ** 0.5 / np.sum(true_pxt ** 2) ** 0.5))
    log.close()

    real_g = x - 0.1
    real_h = x ** 2 / 4

    if smooth_p:
        update_pxt = np.copy(smooth_pxt)
    else:
        update_pxt = np.copy(noisy_pxt)

    true_data = PxtData_NG(t=t, x=x, data=true_pxt)
    noisy_data = PxtData_NG(t=t, x=x, data=noisy_pxt)
    smooth_data = PxtData_NG(t=t, x=x, data=smooth_pxt)
    update_data = PxtData_NG(t=t, x=x, data=update_pxt)

    # end 2 end
    true_data.sample_train_split_e2e(test_range=test_range)
    noisy_data.sample_train_split_e2e(test_range=test_range)
    smooth_data.sample_train_split_e2e(test_range=test_range)
    update_data.sample_train_split_e2e(test_range=test_range)

    lsq = FPLeastSquare_NG(x_coord=x, t_sro=t_sro)

    if smooth_p:
        lsq_g, lsq_h, _, _ = lsq.lsq_wo_t(pxt=smooth_data.train_data, t=smooth_data.train_t)
    else:
        lsq_g, lsq_h, _, _ = lsq.lsq_wo_t(pxt=noisy_data.train_data, t=noisy_data.train_t)

    t_lsq_g, t_lsq_h, dt, p_mat = lsq.lsq_wo_t(pxt=true_data.train_data, t=true_data.train_t)

    # ===========================
    dx = PDeGrid.pde_1d_mat(7, 1, x_points) * x_points / (x_max - x_min)
    dxx = PDeGrid.pde_1d_mat(7, 2, x_points) * x_points ** 2 / (x_max - x_min) ** 2
    true_data_grid = PxtData(t_gap=t_gap, x=x, data=true_pxt)
    # print(np.sum((dx - lsq.dx) ** 2), np.sum((dxx - lsq.dxx) ** 2))
    fpe_lsq = FPLeastSquare(x_points=x_points, dx=dx, dxx=dxx)

    true_data_grid.sample_train_split_e2e(valid_ratio=0, test_ratio=0.1, recur_win=recur_win_p)
    t_lsq_x, t_lsq_y = true_data_grid.process_for_lsq_wo_t()
    t_lsq_g1, t_lsq_h1, t_p_mat, t_abh_mat, t_dev_hh = fpe_lsq.lsq_wo_t(t_lsq_x, t_lsq_y)

    # plt.figure()
    # plt.plot(x, t_lsq_g1, 'r*')
    # plt.plot(x, t_lsq_g, 'b+')
    # plt.plot(x, real_g, 'k')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(x, t_lsq_h1, 'r*')
    # plt.plot(x, t_lsq_h, 'b+')
    # plt.plot(x, real_h, 'k')
    # plt.show()
    # sys.exit()

    if gh == 'real':
        gg_v, hh_v = real_g, real_h
    else:
        gg_v, hh_v = lsq_g, lsq_h

    gg_v = np.expand_dims(gg_v, axis=-1)
    gg_v = np.expand_dims(gg_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)

    fpe_net_ng = FPENet_NG(x_coord=x, name=name, t_sro=t_sro)
    gh_nn_ng = fpe_net_ng.recur_train_gh(learning_rate=learning_rate_gh, loss=Loss.sum_square)
    p_nn_ng = fpe_net_ng.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square,
                                              fix_g=gg_v, fix_h=hh_v)

    # ===========
    gh_net = FPNetReCur(x_points, name=name, t_gap=t_gap, recur_win=recur_win_gh, dx=dx, dxx=dxx)
    gh_nn = gh_net.recur_train_gh(learning_rate=learning_rate_gh, loss=Loss.sum_square)
    p_net = FPNetReCur(x_points, name=name, t_gap=t_gap, recur_win=recur_win_p, dx=dx, dxx=dxx)
    p_nn = p_net.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square, fix_g=gg_v, fix_h=hh_v)
    # ===========

    train_p_x = np.ones((1, x_points, 1))

    p_weight = noisy_data.train_data.sum(axis=0).sum(axis=0)
    p_weight /= sum(p_weight)
    np.save(directory + '/p_weight.npy', p_weight)

    win_x, win_t, win_y, _ = PxtData_NG.get_recur_win_e2e(true_data.train_data, true_data.train_t,
                                                          recur_win_gh)  # check !!
    train_gh_x = np.copy(win_x)
    train_gh_y = np.copy(win_y)
    train_gh_t = np.copy(win_t)
    win_x, win_t, win_y, win_id = PxtData_NG.get_recur_win_e2e(smooth_data.train_data, smooth_data.train_t,  # check !!
                                                               recur_win_p)
    train_p_p = np.copy(win_x)
    train_p_y = np.copy(win_y)
    train_p_t = np.copy(win_t)
    # print(win_y[0, ...])
    # print(win_t[0])
    # sys.exit()
    win_x, win_y, win_id, win_dt = PxtData.get_recur_win_e2e(smooth_data.train_data, recur_win_p, t_gap)
    # print(win_y[0, 0, :])
    # print(win_dt[0])
    # print(win_y[1, 0, :])
    # print(win_dt[1])
    # print(win_y[2, 0, :])
    # print(win_dt[2])
    # print(win_y[3, 0, :])
    # print(win_dt[3])
    # print(win_y[4, 0, :])
    # print(win_dt[4])
    # print(win_y[5, 0, :])
    # print(win_dt[5])
    # print(win_y[45, 0, :])
    # print(win_dt[45])
    # print(win_y[44, 0, :])
    # print(win_dt[44])
    # print(win_y[43, 0, :])
    # print(win_dt[43])
    # print(win_y[42, 0, :])
    # print(win_dt[42])
    # print(win_y[41, 0, :])
    # print(win_dt[41])
    # print(win_y[94, 0, :])
    # print(win_dt[94])
    # print(win_dt)
    # sys.exit()

    log = open(directory + '/train.log', 'a')
    true_train_x, _, _, _ = PxtData_NG.get_recur_win_e2e(true_data.train_data, true_data.train_t, recur_win_p)
    log.write('Initial error of p: {} \n'.format(np.sum((train_p_p - true_train_x) ** 2) ** 0.5))
    log.close()
    print(true_data.train_data.shape, true_train_x.shape)

    # dyn_learning_rate_p = learning_rate_p
    for iter_ in range(n_iter):
        log = open(directory + '/train.log', 'a')
        log.write('Iter: {} \n'.format(iter_))

        # train gh
        # gh_nn.get_layer(name=name + 'g').set_weights([gg_v])
        # gh_nn.get_layer(name=name + 'h').set_weights([hh_v])
        # gh_nn_ng.get_layer(name=name + 'g').set_weights([gg_v])
        # gh_nn_ng.get_layer(name=name + 'h').set_weights([hh_v])
        #
        # es = callbacks.EarlyStopping(verbose=verb, patience=patience)
        # gh_nn.fit(train_gh_x, train_gh_y, epochs=gh_epoch, batch_size=64, verbose=verb, callbacks=[es],
        #           validation_split=0.2)
        #
        # gg_v = gh_nn.get_layer(name=name + 'g').get_weights()[0]
        # hh_v = gh_nn.get_layer(name=name + 'h').get_weights()[0]
        #
        # gh_nn_ng.fit([train_gh_x, train_gh_t], train_gh_y, epochs=gh_epoch, batch_size=64, verbose=verb, callbacks=[es],
        #              validation_split=0.2)
        #
        # gg_v_ng = gh_nn_ng.get_layer(name=name + 'g').get_weights()[0]
        # hh_v_ng = gh_nn_ng.get_layer(name=name + 'h').get_weights()[0]
        #
        # plt.figure()
        # plt.plot(x, gg_v[:, 0, 0], 'r*')
        # plt.plot(x, gg_v_ng[:, 0, 0], 'b+')
        # plt.plot(x, real_g, 'k')
        # plt.show()
        #
        # plt.figure()
        # plt.plot(x, hh_v[:, 0, 0], 'r*')
        # plt.plot(x, hh_v_ng[:, 0, 0], 'b+')
        # plt.plot(x, real_h, 'k')
        # plt.show()
        #
        # np.save(directory + '/gg_iter{}_smooth.npy'.format(iter_), gg_v[:, 0, 0])
        # np.save(directory + '/hh_iter{}_smooth.npy'.format(iter_), hh_v[:, 0, 0])
        # sys.exit()

        # y_model_ng = gh_nn_ng.predict([train_gh_x[:], train_gh_t[:]])
        # y_model = gh_nn.predict(train_gh_x[:])
        #
        # print(train_gh_t[10:11])
        # dt = gh_nn.get_layer(name=name + 'dt').get_weights()[0]
        # print(dt)
        # print(y_model_ng.shape, y_model.shape)
        # for i in range(y_model_ng.shape[0]):
        #     print(i, np.sum((y_model[i] - train_gh_y[i])**2))
        # print(eeer.shape)
        # print(np.sum((y_model[2] - train_gh_y[10:11]) ** 2))
        # print(np.sum((y_model - train_gh_y[:]) ** 2))
        # print(y_model[0][0, :10], y_model[1][0, :10], y_model[2][0, :10])
        # print(y_model_ng[0][0, :10], y_model_ng[1][0, :10], y_model_ng[2][0, :10])
        # sys.exit()

        # print('Shape of y_model:', y_model.shape)
        # print('Error from gh training', np.sum((gg_v[:, 0, 0] - real_g[:]) ** 2),
        #       np.sum((hh_v[:, 0, 0] - real_h[:]) ** 2),
        #       np.sum((train_gh_y - y_model) ** 2))
        # log.write('gh training: {}\n'.format(np.sum((train_gh_y - y_model) ** 2)))
        # log.write('Ratio Error of g: {}, h: {}\n'.format(np.sum((gg_v[:, 0, 0] - real_g) ** 2) / np.sum(real_g ** 2),
        #                                                  np.sum((hh_v[:, 0, 0] - real_h) ** 2) / np.sum(real_h ** 2)))
        # log.write('Weighted Error of g: {}, h: {}\n'.format(
        #     np.sum(p_weight * (gg_v[:, 0, 0] - real_g) ** 2) / np.sum(p_weight * real_g ** 2),
        #     np.sum(p_weight * (hh_v[:, 0, 0] - real_h) ** 2) / np.sum(p_weight * real_h ** 2)))
        # log.write('Error of g: {}, h: {} \n'.format(np.sum((gg_v[:, 0, 0] - real_g) ** 2),
        #                                             np.sum((hh_v[:, 0, 0] - real_h) ** 2)))
        # log.close()

        n_sample = train_p_p.shape[0]

        # train p
        p_nn.get_layer(name=name + 'g').set_weights([gg_v])
        p_nn.get_layer(name=name + 'h').set_weights([hh_v])
        p_nn_ng.get_layer(name=name + 'g').set_weights([gg_v])
        p_nn_ng.get_layer(name=name + 'h').set_weights([hh_v])

        # total_train_p_loss_before = 0
        # total_train_p_loss_after = 0
        # print(update_data.train_data.shape)
        test_p = np.zeros(update_data.train_data.shape)
        test_p_ng = np.zeros(update_data.train_data.shape)

        for sample in range(n_sample):
            sample_id, t_id = win_id[sample]  # no true data, end2end
            es = callbacks.EarlyStopping(verbose=verb, patience=patience)
            print('Training P, Sample id: {}, time id {}'.format(sample_id, t_id))

            p_nn.get_layer(name=name + 'p').set_weights([train_p_p[sample].reshape(-1, 1, 1)])
            p_nn.get_layer(name=name + 'dt').set_weights([win_dt[sample].reshape(-1, 1, recur_win_p)])
            p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p[sample].reshape(-1, 1, 1)])

            y_model_ng = p_nn_ng.predict([train_p_x, train_p_t[sample:sample + 1, :, :]])
            y_model = p_nn.predict(train_p_x)

            dt = p_nn.get_layer(name=name + 'dt').get_weights()[0]

            if np.sum((train_p_t[sample:sample + 1, :, :] - dt)**2) > 0.0000000000000001:
                print(sample, np.sum((train_p_t[sample:sample + 1, :, :] - dt)**2))
                print(train_p_t[sample:sample + 1, :, :])
                print(dt)
                sys.exit()

            # print(y_model.shape, y_model.shape)
            # print(y_model_ng.shape, y_model_ng.shape)
            # print(y_model[0][0, :5, :5])
            # print(y_model[1][0, :5, :5])
            # print(y_model[2][0, :5, :5])
            # print(y_model_ng[0][0, :5, :5])
            # print(y_model_ng[1][0, :5, :5])
            # print(y_model_ng[2][0, :5, :5])
            # sys.exit()

            # p_loss_before = p_nn.evaluate(train_p_x, train_p_y[sample:sample + 1, :, :])
            p_nn.fit(train_p_x, train_p_y[sample:sample+1, :, :],
                     epochs=200, verbose=verb, callbacks=[es],
                     validation_data=[train_p_x, train_p_y[sample:sample+1, :, :]])
            # p_loss_after = p_nn.evaluate(train_p_x, train_p_y[sample:sample+1, :, :])

            # print(p_loss_before, p_loss_after)

            test_p[sample_id, t_id] = p_nn.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]

            # ============================

            # p_loss_before = p_nn_ng.evaluate([train_p_x, train_p_t[sample:sample+1, :, :]],
            #                                  train_p_y[sample:sample + 1, :, :])
            p_nn_ng.fit([train_p_x, train_p_t[sample:sample+1, :, :]], train_p_y[sample:sample+1, :, :],
                        epochs=200, verbose=verb, callbacks=[es],
                        validation_data=[[train_p_x, train_p_t[sample:sample+1, :, :]],
                        train_p_y[sample:sample+1, :, :]])

            test_p_ng[sample_id, t_id] = p_nn_ng.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            # p_loss_after = p_nn_ng.evaluate([train_p_x, train_p_t[sample:sample+1, :, :]],
            #                                 train_p_y[sample:sample + 1, :, :])
            # print(p_loss_before, p_loss_after)

            # sys.exit()
            # p_nn_ng.get_layer(name=name + 'p').set_weights([train_p_p[sample].reshape(-1, 1, 1)])
            # p_loss_ng = p_nn_ng.evaluate([train_p_x, train_p_t[sample:sample + 1, :, :]],
            #                              train_p_y[sample:sample + 1, :, :])
            # print(sample, p_loss, p_loss_ng)

            # ============

        print(np.sum((smooth_data.train_data - true_data.train_data)**2))
        print(np.sum((test_p - true_data.train_data)**2))
        print(np.sum((test_p_ng - true_data.train_data) ** 2))
        np.savez_compressed(directory + '/train_p', p=test_p, p_nn_ng=test_p_ng)


        # update_data.train_data = padding_by_axis2_smooth(update_data.train_data, 5)
        # win_x, win_y, win_t, _ = PxtData_NG.get_recur_win_e2e(update_data.train_data, update_data.train_t, recur_win_gh)
        # train_gh_x = np.copy(win_x)  # not end2end
        # train_gh_y = np.copy(win_y)
        #
        # win_x, win_y, win_t, _ = PxtData_NG.get_recur_win_e2e(update_data.train_data, update_data.train_t, recur_win_p)
        # train_p_p = np.copy(win_x)
        #
        # log = open(directory + '/train.log', 'a')
        # log.write('Total error of p training before: {}, after: {}\n'.format(total_train_p_loss_before,
        #                                                                      total_train_p_loss_after))
        # log.write('Error to true p: {} ratio {}, Error to noisy p: {} ratio {} \n'.
        #           format(np.sum((train_p_p - true_train_x) ** 2) ** 0.5,
        #                  (np.sum((train_p_p - true_train_x) ** 2) / np.sum(true_train_x ** 2)) ** 0.5,
        #                  np.sum((train_p_p - true_train_x) ** 2) ** 0.5,
        #                  (np.sum((train_p_p - true_train_x) ** 2) / np.sum(true_train_x ** 2)) ** 0.5))
        #
        # predict_test_euler = test_steps(x, gg_v[:, 0, 0], hh_v[:, 0, 0], update_data)
        #
        # log.write('To True data, steps: \t')
        # log.write('To True data, euler: \t')
        # for pos in range(test_range):
        #     log.write('{} \t'.format(np.sum((predict_test_euler[:, pos, :] - true_data.test_data[:, pos, :]) ** 2)))
        # log.write('\n')
        # log.write('To Noisy data, euler: \t')
        # for pos in range(test_range):
        #     log.write('{} \t'.format(np.sum((predict_test_euler[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2)))
        # log.write('\n')
        # log.close()
        #
        # # save
        # np.save(directory + '/gg_iter{}_smooth.npy'.format(iter_), gg_v[:, 0, 0])
        # np.save(directory + '/hh_iter{}_smooth.npy'.format(iter_), hh_v[:, 0, 0])
        # np.save(directory + '/train_data_iter{}.npy'.format(iter_), update_data.train_data)


if __name__ == '__main__':
    main(run_id=config.RUN_ID, p_patience=10, smooth_gh=0.1, smooth_p=True)
