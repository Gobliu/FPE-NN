import numpy as np
from scipy import signal
# import OU_config as config  # B_config or OU_config
import Boltz_config as config
import sys
import os
from keras import callbacks, backend, losses
import matplotlib.pyplot as plt
sys.path.insert(1, '../GridModules')
from PartialDerivativeGrid import PartialDerivativeGrid as PDeGrid
from PxtData import PxtData
from Loss import Loss
from FPLeastSquare import FPLeastSquare
from FPNet_ReCur import FPNetReCur
from GaussianSmooth import GaussianSmooth
from FokkerPlankEqn import FokkerPlankForward as FPF

np.set_printoptions(suppress=True)

name = 'NoisyOU'
seed = config.SEED
x_min = config.X_MIN
x_max = config.X_MAX
x_points = config.X_POINTS
x_gap = (x_max - x_min) / x_points
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
gh = 'lsq'
n_iter = 1000
valid_ratio = 0
test_ratio = 0.1
# test_ratio = 0.2    # for ID 7
sf_range = 7
t_range = [0, 100]


def test_steps(g, h, data):
    n_sample = data.test_data.shape[0]
    predict_t_points = data.test_data.shape[1] + 1
    predict_pxt_step = np.zeros((n_sample, predict_t_points, x_points))
    predict_pxt_euler = np.zeros((n_sample, predict_t_points, x_points))
    for sample in range(data.n_sample):
        p0 = data.train_data[sample, -1, :]
        predict_pxt_step[sample] = FPF.ghx2pxt_pad_smooth(g, h, p0, x_gap, t_gap, predict_t_points, direction=1)
        predict_pxt_euler[sample] = FPF.ghx2pxt_euler(g, h, p0, x_gap, t_gap, predict_t_points, direction=1)
    op_pxt_step = predict_pxt_step[:, 1:, :]
    op_pxt_euler = predict_pxt_euler[:, 1:, :]
    return op_pxt_step, op_pxt_euler


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
        directory = './Result/Boltz/id{}_{}_p{}_win{}{}'.format(run_id, run_, p_patience, recur_win_gh, recur_win_p)
        if os.path.exists(directory):
            run_ += 1
            pass
        else:
            os.makedirs(directory)
            break
    dx = PDeGrid.pde_1d_mat(7, 1, x_points) * x_points / (x_max - x_min)
    dxx = PDeGrid.pde_1d_mat(7, 2, x_points) * x_points ** 2 / (x_max - x_min) ** 2

    # load = np.load('./Pxt/pseudoB/B_OU_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma))
    # load = np.load('./Pxt/OU/OU_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma))
    load = np.load('./Pxt/Bessel/B_f_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma))
    # load = np.load('./Pxt/Boltz/Boltz_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma))
    x = load[0, 0, :]
    true_pxt = load[:, 1:, :]
    true_pxt = true_pxt[:, t_range[0]: t_range[1], :]
    # load = np.load('./Pxt/pseudoB/B_OU_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma))
    # load = np.load('./Pxt/OU/OU_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma))
    load = np.load('./Pxt/Bessel/B_f_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma))
    # load = np.load('./Pxt/Boltz/Boltz_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma))
    noisy_pxt = load[:, 1:, :]
    noisy_pxt = noisy_pxt[:, t_range[0]: t_range[1], :]

    # data = np.load('./Pxt/Boltz_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
    # x = data['x']
    # # x_points = x.shape[0]
    # # print(x_points)
    # # t = data['t']
    # true_pxt = data['true_pxt']
    # noisy_pxt = data['noisy_pxt']

    true_pxt[true_pxt < 0] = 0
    noisy_pxt[noisy_pxt < 0] = 0

    log = open(directory + '/train.log', 'w')
    log.write('{}_pxt_{}_sigma{}.npy \n'.format(run_id, seed, sigma))
    log.write('{}_noisy_{}_sigma{}.npy \n'.format(run_id, seed, sigma))
    log.write('learning rate gh: {} \n'.format(learning_rate_gh))
    log.write('learning rate p: {} \n'.format(learning_rate_p))
    log.write('p_epoch_factor {}, sf_range: {} \n'.format(p_epoch_factor, sf_range))
    log.write('Initial error of pxt before smooth: {} ratio {}\n'.
              format(np.sum((noisy_pxt - true_pxt)**2)**0.5,
                     np.sum((noisy_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))

    smooth_pxt = padding_by_axis2_smooth(noisy_pxt, 5)
    log.write('Initial error of pxt after smooth: {} ratio {}\n'.
              format(np.sum((smooth_pxt - true_pxt)**2)**0.5,
                     np.sum((smooth_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))
    log.close()

    # real_g = 2.86 * x
    # real_g = 1/x - 0.2
    # real_g = 1 / x + 0.002
    # real_h = 0.0013 * np.ones(x_points)
    # real_h = 0.0015 * np.ones(x_points)
    # real_h = 0.5 * np.ones(x_points)
    # real_h = 0.00005 * np.ones(x_points)

    real_g = x - 0.1
    # real_h = x ** 2 * 0.1 / 2
    real_h = x ** 2 / 4

    true_data = PxtData(t_gap=t_gap, x=x, data=true_pxt)
    noisy_data = PxtData(t_gap=t_gap, x=x, data=noisy_pxt)
    smooth_data = PxtData(t_gap=t_gap, x=x, data=smooth_pxt)

    # end 2 end
    true_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)
    noisy_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)
    smooth_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)
    if smooth_p:
        update_pxt = np.copy(smooth_pxt)
    else:
        update_pxt = np.copy(noisy_pxt)
    update_data = PxtData(t_gap=t_gap, x=x, data=update_pxt)
    update_data.sample_train_split_e2e(valid_ratio=valid_ratio, test_ratio=test_ratio, recur_win=recur_win_p)

    if smooth_p:
        lsq_x, lsq_y = smooth_data.process_for_lsq_wo_t()
    else:
        lsq_x, lsq_y = noisy_data.process_for_lsq_wo_t()
    fpe_lsq = FPLeastSquare(x_points=x_points, dx=dx, dxx=dxx)
    print(lsq_x.shape, lsq_y.shape)

    lsq_g, lsq_h, p_mat, abh_mat, dev_hh = fpe_lsq.lsq_wo_t(lsq_x, lsq_y)

    t_lsq_x, t_lsq_y = true_data.process_for_lsq_wo_t()
    t_lsq_g, t_lsq_h, t_p_mat, t_abh_mat, t_dev_hh = fpe_lsq.lsq_wo_t(t_lsq_x, t_lsq_y)

    if gh == 'real':
        gg_v, hh_v = real_g, real_h
    else:
        gg_v, hh_v = lsq_g, lsq_h

    plt.figure()
    plt.plot(x, lsq_g, 'r+')
    plt.plot(x, t_lsq_g, 'b*')
    plt.plot(x, real_g, 'k')
    plt.show()

    plt.figure()
    plt.plot(x, lsq_h, 'r+')
    plt.plot(x, t_lsq_h, 'b*')
    plt.plot(x, real_h, 'k')
    plt.show()
    # sys.exit()

    gg_v = np.expand_dims(gg_v, axis=-1)
    gg_v = np.expand_dims(gg_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)
    hh_v = np.expand_dims(hh_v, axis=-1)

    gh_net = FPNetReCur(x_points, name=name, t_gap=t_gap, recur_win=recur_win_gh, dx=dx, dxx=dxx)
    gh_nn = gh_net.recur_train_gh(learning_rate=learning_rate_gh, loss=Loss.sum_square)
    # gh_nn = gh_net.recur_train_gh(learning_rate=learning_rate_gh, loss=losses.mean_absolute_error)
    p_net = FPNetReCur(x_points, name=name, t_gap=t_gap, recur_win=recur_win_p, dx=dx, dxx=dxx)
    p_nn = p_net.recur_train_p_direct(learning_rate=learning_rate_p, loss=Loss.sum_square, fix_g=gg_v, fix_h=hh_v)
    # p_nn = p_net.recur_train_p_direct(learning_rate=learning_rate_p, loss=losses.mean_absolute_error,
    #                                   fix_g=gg_v, fix_h=hh_v)

    train_p_x = np.ones((1, x_points, 1))
    true_data.process_for_recur_net_e2e(recur_win=recur_win_p)
    smooth_data.process_for_recur_net_e2e(recur_win=recur_win_p)
    noisy_data.process_for_recur_net_e2e(recur_win=recur_win_p)

    p_weight = noisy_data.train_data.sum(axis=0).sum(axis=0)
    p_weight /= sum(p_weight)
    np.save(directory + '/p_weight.npy', p_weight)

    # train gh not end2end, train p end2end
    if smooth_p:
        win_x, win_y, win_id = PxtData.get_recur_win(smooth_data.train_data, recur_win_gh)
        train_gh_x = np.copy(win_x)
        train_gh_y = np.copy(win_y)
        win_x, win_y, win_id, win_dt = PxtData.get_recur_win_e2e(smooth_data.train_data, recur_win_p, t_gap)
        train_p_p = np.copy(win_x)
        train_p_y = np.copy(win_y)
    else:
        win_x, win_y, win_id = PxtData.get_recur_win(noisy_data.train_data, recur_win_gh)
        train_gh_x = np.copy(win_x)
        train_gh_y = np.copy(win_y)
        win_x, win_y, win_id, win_dt = PxtData.get_recur_win_e2e(noisy_data.train_data, recur_win_p, t_gap)
        train_p_p = np.copy(win_x)
        train_p_y = np.copy(win_y)

    log = open(directory + '/train.log', 'a')
    log.write('Initial error of p: {} \n'.format(np.sum((train_p_p - true_data.train_x)**2)**0.5))
    log.close()

    dyn_learning_rate_p = learning_rate_p
    for iter_ in range(n_iter):
        log = open(directory + '/train.log', 'a')
        log.write('Iter: {} \n'.format(iter_))

        # smooth
        # if iter_ < 30:
        gg_v[:, 0, 0] = GaussianSmooth.gaussian1d(gg_v[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))
        hh_v[:, 0, 0] = GaussianSmooth.gaussian1d(hh_v[:, 0, 0], sigma=1 / (smooth_gh * iter_+1))
        # gg_v[:, 0, 0] = signal.savgol_filter(gg_v[:, 0, 0], sf_range, 2)
        # hh_v[:, 0, 0] = signal.savgol_filter(hh_v[:, 0, 0], sf_range, 2)

        # train gh
        gh_nn.get_layer(name=name + 'g').set_weights([gg_v])
        gh_nn.get_layer(name=name + 'h').set_weights([hh_v])

        # backend.set_value(gh_nn.optimizer.lr, learning_rate_gh / (10**(iter_//100)))
        es = callbacks.EarlyStopping(verbose=verb, patience=patience)
        gh_nn.fit(train_gh_x, train_gh_y, epochs=gh_epoch, batch_size=64, verbose=verb, callbacks=[es],
                  validation_split=0.2)

        gg_v = gh_nn.get_layer(name=name + 'g').get_weights()[0]
        hh_v = gh_nn.get_layer(name=name + 'h').get_weights()[0]

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

        n_sample = train_p_p.shape[0]

        # train p
        p_nn.get_layer(name=name + 'g').set_weights([gg_v])
        p_nn.get_layer(name=name + 'h').set_weights([hh_v])
        backend.set_value(p_nn.optimizer.lr, dyn_learning_rate_p)

        total_train_p_loss_before = 0
        total_train_p_loss_after = 0
        for sample in range(n_sample):
            sample_id, t_id = smooth_data.train_id[sample]      # no true data, end2end
            print('Training P, Sample id: {}, time id {}'.format(sample_id, t_id))
            p_nn.get_layer(name=name + 'p').set_weights([train_p_p[sample].reshape(-1, 1, 1)])
            p_nn.get_layer(name=name + 'dt').set_weights([smooth_data.train_dt[sample].reshape(-1, 1, recur_win_p)])
            es = callbacks.EarlyStopping(verbose=verb, patience=p_patience)
            # May 10 change iter_//5
            p_loss = p_nn.evaluate(train_p_x, train_p_y[sample:sample+1, :, :])
            total_train_p_loss_before += p_loss
            p_nn.fit(train_p_x, train_p_y[sample:sample+1, :, :], epochs=iter_//p_epoch_factor + 1, verbose=verb,
                     callbacks=[es], validation_data=[train_p_x, train_p_y[sample:sample+1, :, :]])

            update_data.train_data[sample_id, t_id] = p_nn.get_layer(name=name + 'p').get_weights()[0][:, 0, 0]
            p_loss = p_nn.evaluate(train_p_x, train_p_y[sample:sample+1, :, :])
            total_train_p_loss_after += p_loss

        update_data.train_data = padding_by_axis2_smooth(update_data.train_data, 5)
        win_x, win_y, win_id = PxtData.get_recur_win(update_data.train_data, recur_win_gh)
        train_gh_x = np.copy(win_x)  # not end2end
        train_gh_y = np.copy(win_y)

        win_x, win_y, win_id, win_dt = PxtData.get_recur_win_e2e(update_data.train_data, recur_win_p, t_gap)
        train_p_p = np.copy(win_x)

        log = open(directory + '/train.log', 'a')
        log.write('Total error of p training before: {}, after: {}\n'.format(total_train_p_loss_before,
                                                                             total_train_p_loss_after))
        log.write('Error to true p: {} ratio {}, Error to noisy p: {} ratio {} \n'.
                  format(np.sum((train_p_p - true_data.train_x)**2)**0.5,
                         (np.sum((train_p_p - true_data.train_x) ** 2)/np.sum(true_data.train_x**2))**0.5,
                         np.sum((train_p_p - noisy_data.train_x) ** 2)**0.5,
                         (np.sum((train_p_p - noisy_data.train_x) ** 2)/np.sum(noisy_data.train_x**2))**0.5))

        predict_test_steps, predict_test_euler = test_steps(gg_v[:, 0, 0], hh_v[:, 0, 0], update_data)

        test_t_points = predict_test_steps.shape[1]
        log.write('To True data, steps: \t')
        for pos in range(test_t_points):
            log.write('{} \t'.format(np.sum((predict_test_steps[:, pos, :] - true_data.test_data[:, pos, :]) ** 2)))
        log.write('\n')
        log.write('To True data, euler: \t')
        for pos in range(test_t_points):
            log.write('{} \t'.format(np.sum((predict_test_euler[:, pos, :] - true_data.test_data[:, pos, :]) ** 2)))
        log.write('\n')
        log.write('To Noisy data, steps: \t')
        for pos in range(test_t_points):
            log.write('{} \t'.format(np.sum((predict_test_steps[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2)))
        log.write('\n')
        log.write('To Noisy data, euler: \t')
        for pos in range(test_t_points):
            log.write('{} \t'.format(np.sum((predict_test_euler[:, pos, :] - noisy_data.test_data[:, pos, :]) ** 2)))
        log.write('\n')
        log.close()

        # save
        np.save(directory + '/gg_iter{}_smooth.npy'.format(iter_), gg_v[:, 0, 0])
        np.save(directory + '/hh_iter{}_smooth.npy'.format(iter_), hh_v[:, 0, 0])
        np.save(directory + '/train_data_iter{}.npy'.format(iter_), update_data.train_data)


if __name__ == '__main__':
    main(run_id=config.RUN_ID, p_patience=10, smooth_gh=0.1, smooth_p=True)
