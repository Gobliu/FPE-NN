import numpy as np
from scipy import signal
# import OU_config as config  # B_config or OU_config
# import B_config as config
import Boltz_config as config
import sys
import os
from keras import callbacks, backend, losses
import matplotlib.pyplot as plt
sys.path.insert(1, './ifpe-modules')
from PartialDerivativeGrid import PartialDerivativeGrid as PDeGrid
from PxtData import PxtData
from Loss import Loss
from FPLeastSquare import FPLeastSquare
from FPNet_ReCur import FPNetReCur
from GaussianSmooth import GaussianSmooth
from FokkerPlankEqn import FokkerPlankForward as FPF

np.set_printoptions(suppress=True)

name = 'NoisyB'
seed = config.SEED
x_min = config.X_MIN
x_max = config.X_MAX
x_points = config.X_POINTS
x_gap = (x_max - x_min) / x_points
# t_gap = config.T_GAP
# t_gap = 0.001
sigma = config.SIGMA
learning_rate_gh = config.LEARNING_RATE_GH
learning_rate_p = config.LEARNING_RATE_P
gh_epoch = config.EPOCH
p_epoch = 1
patience = config.PATIENCE
batch_size = config.BATCH_SIZE
recur_win_gh = 5
recur_win_p = 13
verb = 2
p_epoch_factor = 5
gh = 'lsq'
n_iter = 1000
valid_ratio = 0
test_ratio = 0.1
# test_ratio = 0.2    # for ID 7
t_range = [0, 50]


def main(run_id, p_patience):
    run_ = 0
    while run_ < 100:
        # directory = './Result/gh_only/Bessel/id{}_p{}_win{}_{}'.format(run_id, p_patience, recur_win_gh, run_)
        directory = './Result/gh_only/Boltz/id{}_p{}_win{}_{}'.format(run_id, p_patience, recur_win_gh, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break
    dx = PDeGrid.pde_1d_mat(7, 1, x_points) * x_points / (x_max - x_min)
    dxx = PDeGrid.pde_1d_mat(7, 2, x_points) * x_points ** 2 / (x_max - x_min) ** 2

    # # load = np.load('./Pxt/pseudoB/B_OU_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma))
    # # load = np.load('./Pxt/OU/OU_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma))
    # load = np.load('./Pxt/Bessel/B_f_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma))
    # x = load[0, 0, :]
    # true_pxt = load[:, 1:, :]
    # true_pxt = true_pxt[:, t_range[0]: t_range[1], :]
    load = np.load('./Pxt/Boltz_1.npz')
    x = load['x']
    t = load['t']
    print(t)
    t_gap = t[0, 1] - t[0, 0]
    print(t_gap)
    true_pxt = load['true_pxt']

    true_pxt[true_pxt < 0] = 0
    # noisy_pxt[noisy_pxt < 0] = 0

    log = open(directory + '/train.log', 'w')
    log.write('{}_pxt_{}_sigma{}.npy \n'.format(run_id, seed, sigma))
    log.write('{}_noisy_{}_sigma{}.npy \n'.format(run_id, seed, sigma))
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    # log.write('Initial error of pxt before smooth: {} ratio {}\n'.
    #           format(np.sum((noisy_pxt - true_pxt)**2)**0.5,
    #                  np.sum((noisy_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))

    # smooth_pxt = padding_by_axis2_smooth(noisy_pxt, 5)
    # log.write('Initial error of pxt after smooth: {} ratio {}\n'.
    #           format(np.sum((smooth_pxt - true_pxt)**2)**0.5,
    #                  np.sum((smooth_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))
    log.close()

    # real_g = 2.86 * x
    # real_h = 0.0013 * np.ones(x_points)

    # real_g = 1/x - 0.2
    # real_h = 0.5 * np.ones(x_points)

    real_g = x - 0.1
    real_h = x**2 * 0.1 / 2

    true_data = PxtData(t_gap=t_gap, x=x, data=true_pxt)
    # noisy_data = PxtData(t_gap=t_gap, x=x, data=noisy_pxt)
    # smooth_data = PxtData(t_gap=t_gap, x=x, data=smooth_pxt)

    # end 2 end
    true_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)
    # noisy_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)
    # smooth_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)
    # if smooth_p:
    #     update_pxt = np.copy(smooth_pxt)
    # else:
    #     update_pxt = np.copy(noisy_pxt)
    # update_data = PxtData(t_gap=t_gap, x=x, data=update_pxt)
    # update_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)

    # if smooth_p:
    #     lsq_x, lsq_y = smooth_data.process_for_lsq_wo_t()
    # else:
    #     lsq_x, lsq_y = noisy_data.process_for_lsq_wo_t()
    fpe_lsq = FPLeastSquare(x_points=x_points, dx=dx, dxx=dxx)
    # print(lsq_x.shape, lsq_y.shape)
    #
    # lsq_g, lsq_h, p_mat, abh_mat, dev_hh = fpe_lsq.lsq_wo_t(lsq_x, lsq_y)

    t_lsq_x, t_lsq_y = true_data.process_for_lsq_wo_t()
    t_lsq_g, t_lsq_h, t_p_mat, t_abh_mat, t_dev_hh = fpe_lsq.lsq_wo_t(t_lsq_x, t_lsq_y)

    if gh == 'real':
        gg_v, hh_v = real_g, real_h
    else:
        gg_v, hh_v = t_lsq_g, t_lsq_h

    plt.figure()
    # plt.plot(x, lsq_g, 'r')
    plt.plot(x, t_lsq_g, 'b^')
    plt.plot(x, real_g, 'k')
    plt.show()

    # print(real_h)
    plt.figure()
    # plt.plot(x, lsq_h, 'r')
    plt.plot(x, t_lsq_h, 'b^')
    plt.plot(x, real_h, 'k')
    plt.legend()
    plt.show()
    # sys.exit()

    gg_v = np.expand_dims(gg_v, axis=-1)
    gg_v = np.expand_dims(gg_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)

    gh_net = FPNetReCur(x_points, name=name, t_gap=t_gap, recur_win=recur_win_gh, dx=dx, dxx=dxx)
    gh_nn = gh_net.recur_train_gh(learning_rate=learning_rate_gh, loss=Loss.sum_square)

    p_weight = true_data.train_data.sum(axis=0).sum(axis=0)
    p_weight /= sum(p_weight)
    np.save(directory + '/p_weight.npy', p_weight)

    # train gh not end2end, train p end2end
    win_x, win_y, win_id = PxtData.get_recur_win(true_data.train_data, recur_win_gh)
    train_gh_x = np.copy(win_x)
    train_gh_y = np.copy(win_y)

    log = open(directory + '/train.log', 'a')

    for iter_ in range(1):
        log = open(directory + '/train.log', 'a')
        log.write('Iter: {} \n'.format(iter_))

        # train gh
        gh_nn.get_layer(name=name + 'g').set_weights([gg_v])
        gh_nn.get_layer(name=name + 'h').set_weights([hh_v])

        # backend.set_value(gh_nn.optimizer.lr, learning_rate_gh / (10**(iter_//100)))
        es = callbacks.EarlyStopping(verbose=verb, patience=patience)
        gh_nn.fit(train_gh_x, train_gh_y, epochs=gh_epoch, batch_size=64, verbose=verb, callbacks=[es],
                  validation_split=0.2)

        gg_v = gh_nn.get_layer(name=name + 'g').get_weights()[0]
        hh_v = gh_nn.get_layer(name=name + 'h').get_weights()[0]

        plt.figure()
        plt.plot(x, t_lsq_g, 'b')
        plt.plot(x, gg_v[:, 0, 0], 'r*')
        plt.plot(x, real_g, 'k')
        plt.show()

        print(real_h)
        plt.figure()
        plt.plot(x, t_lsq_h, 'b')
        plt.plot(x, hh_v[:, 0, 0], 'r*')
        plt.plot(x, real_h, 'k')
        plt.legend()
        plt.show()

        print('Diff before and after training', np.sum((gg_v[:, 0, 0] - t_lsq_g[:])**2),
              np.sum((hh_v[:, 0, 0] - t_lsq_h[:])**2))
        y_model = gh_nn.predict(train_gh_x)
        print('Shape of y_model:', y_model.shape)
        print('Error from gh training', np.sum((gg_v[:, 0, 0] - real_g[:])**2),
              np.sum((hh_v[:, 0, 0] - real_h[:])**2),
              np.sum((train_gh_y - y_model)**2))
        log.write('gh training: {}\n'.format(np.sum((train_gh_y - y_model)**2)))
        log.write('Ratio Error of g: {}, h: {}\n'.format(np.sum((gg_v[:, 0, 0] - real_g)**2)/np.sum(real_g**2),
                                                         np.sum((hh_v[:, 0, 0] - real_h)**2)/np.sum(real_h**2)))
        log.write('Weighted Error of g: {}, h: {}\n'.format(
                                    np.sum(p_weight*(gg_v[:, 0, 0] - real_g)**2)/np.sum(p_weight*real_g**2),
                                    np.sum(p_weight*(hh_v[:, 0, 0] - real_h)**2)/np.sum(p_weight*real_h**2)))
        log.write('Error of g: {}, h: {} \n'.format(np.sum((gg_v[:, 0, 0] - real_g)**2),
                                                    np.sum((hh_v[:, 0, 0] - real_h)**2)))
        log.close()

        np.save(directory + '/gg_iter{}_smooth.npy'.format(iter_), gg_v[:, 0, 0])
        np.save(directory + '/hh_iter{}_smooth.npy'.format(iter_), hh_v[:, 0, 0])
        # np.save(directory + '/train_data_iter{}.npy'.format(iter_), update_data.train_data)


if __name__ == '__main__':
    main(run_id=config.RUN_ID, p_patience=10)
