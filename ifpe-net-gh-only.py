import numpy as np
# import OU_config as config
# import B_config as config
import Boltz_config as config
import sys
import os
from keras import callbacks, backend, losses
import matplotlib.pyplot as plt
sys.path.insert(1, './GridModules')
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
t_gap = config.T_GAP
sigma = config.SIGMA
# learning_rate_gh = config.LEARNING_RATE_GH
learning_rate_gh = 0.1 ** 7
learning_rate_p = config.LEARNING_RATE_P
gh_epoch = config.EPOCH
p_epoch = 1
patience = config.PATIENCE
batch_size = config.BATCH_SIZE
recur_win_gh = 3
recur_win_p = 3
verb = 2
p_epoch_factor = 5
gh = 'lsq'
n_iter = 1
valid_ratio = 0
test_ratio = 0.1
# test_ratio = 0.2    # for ID 7
# t_range = [0, 50]


def main(run_id, p_patience):
    run_ = 0
    while run_ < 100:
        directory = '/home/liuwei/GitHub/Result/gh_only/Boltz/id{}_p{}_win{}_{}'.format(run_id, p_patience,
                                                                                        recur_win_gh, run_)
        # directory = '/home/liuwei/GitHub/Result/gh_only/Bessel/id{}_p{}_win{}_{}'.format(run_id, p_patience,
        #                                                                                  recur_win_gh, run_)
        # directory = '/home/liuwei/GitHub/Result/gh_only/OU/id{}_p{}_win{}_{}'.format(run_id, p_patience,
        #                                                                              recur_win_gh, run_)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break
    dx = PDeGrid.pde_1d_mat(7, 1, x_points) * x_points / (x_max - x_min)
    dxx = PDeGrid.pde_1d_mat(7, 2, x_points) * x_points ** 2 / (x_max - x_min) ** 2

    data = np.load('./Pxt/Boltz_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    # data = np.load('./Pxt/Bessel_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    # data = np.load('./Pxt/OU_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    x = data['x']
    true_pxt = data['true_pxt']
    print(true_pxt.shape, x.shape)

    true_pxt[true_pxt < 0] = 0

    log = open(directory + '/train.log', 'w')
    log.write('{}_pxt_{}_sigma{}.npy \n'.format(run_id, seed, sigma))
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.close()

    # Boltz
    real_g = x - 1
    real_h = 0.2 * x**2
    # Bessel
    # real_g = 1/x - 0.2
    # real_h = 0.5 * np.ones(x.shape)
    # OU
    # real_g = 2.86 * x
    # real_h = 0.0013 * np.ones(x.shape)

    true_data = PxtData(t_gap=t_gap, x=x, data=true_pxt)

    true_data.whole_train_split(valid_ratio=0.2, test_ratio=0.2)

    fpe_lsq = FPLeastSquare(x_points=x_points, dx=dx, dxx=dxx)

    t_lsq_x, t_lsq_y = true_data.process_for_lsq_wo_t()
    t_lsq_g, t_lsq_h, t_p_mat, t_abh_mat, t_dev_hh = fpe_lsq.lsq_wo_t(t_lsq_x, t_lsq_y)

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
    print(win_x.shape, win_y.shape)

    win_x, win_y, win_id = PxtData.get_recur_win(true_data.valid_data, recur_win_gh)
    valid_gh_x = np.copy(win_x)
    valid_gh_y = np.copy(win_y)
    print(win_x.shape, win_y.shape)

    win_x, win_y, win_id = PxtData.get_recur_win(true_data.test_data, recur_win_gh)
    test_gh_x = np.copy(win_x)
    test_gh_y = np.copy(win_y)
    print(win_x.shape, win_y.shape)

    log = open(directory + '/train.log', 'a')

    # train gh
    gh_nn.get_layer(name=name + 'g').set_weights([gg_v])
    gh_nn.get_layer(name=name + 'h').set_weights([hh_v])

    y_model = gh_nn.predict(test_gh_x)
    log.write('test loss before training: {}\n'.format(np.sum((test_gh_y - y_model) ** 2)/np.sum(test_gh_y**2)))

    es = callbacks.EarlyStopping(verbose=verb, patience=patience)
    gh_nn.fit(train_gh_x, train_gh_y, epochs=gh_epoch, batch_size=64, verbose=verb, callbacks=[es],
              validation_data=[valid_gh_x, valid_gh_y])

    gg_v = gh_nn.get_layer(name=name + 'g').get_weights()[0]
    hh_v = gh_nn.get_layer(name=name + 'h').get_weights()[0]

    print('Diff before and after training', np.sum((gg_v[:, 0, 0] - t_lsq_g[:])**2),
          np.sum((hh_v[:, 0, 0] - t_lsq_h[:])**2))
    y_model = gh_nn.predict(test_gh_x)
    print('Shape of y_model:', y_model.shape)
    print('Error from gh training', np.sum((gg_v[:, 0, 0] - real_g[:])**2),
          np.sum((hh_v[:, 0, 0] - real_h[:])**2), np.sum((test_gh_y - y_model)**2))
    log.write('test loss after training: {}\n'.format(np.sum((test_gh_y - y_model) ** 2)/np.sum(test_gh_y**2)))
    log.write('Ratio Error of g: {}, h: {}\n'.format(np.sum((gg_v[:, 0, 0] - real_g)**2)/np.sum(real_g**2),
                                                     np.sum((hh_v[:, 0, 0] - real_h)**2)/np.sum(real_h**2)))
    log.write('Error of g: {}, h: {} \n'.format(np.sum((gg_v[:, 0, 0] - real_g)**2),
                                                np.sum((hh_v[:, 0, 0] - real_h)**2)))
    log.write('Error of lsq_g: {}, lsq_h: {} \n'.format(np.sum((t_lsq_g - real_g) ** 2),
                                                        np.sum((t_lsq_h - real_h) ** 2)))

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
    plt.show()

    np.save(directory + '/gg_smooth.npy', gg_v[:, 0, 0])
    np.save(directory + '/hh_smooth.npy', hh_v[:, 0, 0])

    # ========================
    gg_v, hh_v = real_g, real_h
    gg_v = np.expand_dims(gg_v, axis=-1)
    gg_v = np.expand_dims(gg_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    gh_nn.get_layer(name=name + 'g').set_weights([gg_v])
    gh_nn.get_layer(name=name + 'h').set_weights([hh_v])
    y_model = gh_nn.predict(test_gh_x)
    log.write('test loss true gh: {}\n'.format(np.sum((test_gh_y - y_model) ** 2)/np.sum(test_gh_y**2)))
    # ========================
    print(np.sum(test_gh_y**2))

    log.close()


if __name__ == '__main__':
    main(run_id=config.RUN_ID, p_patience=20)
