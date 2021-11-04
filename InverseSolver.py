import os
import sys

import numpy as np
import tensorflow as tf
from scipy import signal, ndimage
from keras import callbacks

from Modules.PxtData import PxtDatagit
from Modules.FPLeastSquare import FPLeastSquare
from Modules.FPENet import FPENet
from Modules.Loss import Loss

np.set_printoptions(suppress=True)

# ~~~~ parameter setting ~~~~
run_id = 0                      # any remark, string or integer
seed = 20211103
verb = 2                        # verbose for tensorflow


ip_dir = './Pxt'                # directory of the input npz
ip_name = 'sinusoid.npz'                    # name of the input npz
op_dir = '/home/liuwei/GitHub/Result'       # directory of the training output, a folder will be auto-generated
model_name = 'Noisy'            # name of the model

gh_learning_rate = 0.1**5       # learning rate of gh trainer
gh_epoch = 200000               # max epoch of gh trainer, usually will stop early
gh_patience = 20                # patience of gh trainer, for early stopping
batch_size = 64                 # batch size for gh training; p training always use batch size 1
gh_recur_win = 5                # how many neighbouring time points used in gh trainer

p_learning_rate = 0.1**5        # learning rate of p trainer
p_epoch_factor = 5              # factor to control the epoch increasing of p trainer
p_patience = 10                 # patience of p trainer, for early stopping
p_recur_win = 5                 # how many neighbouring time points used in p trainer

n_iter = 500                    # how many rounds of alternating training on gh and p
t_sro = 7                       # total sum rules for generating derivative matrix, refer to PDE-Net paper
use_true_gh = False             # use true gh as the initial gh, for testing

smooth_p = True                 # smooth pxt with Savitzky-Golay filter
sf_range = 7                    # point range of Savitzky-Golay filter
smooth_gh = True                # smooth gh with Gaussian filter
smooth_gh_sigma = 10            # initial sigma in Gaussian filter to smooth gh

# ~~~~ setting ends ~~~~


def sf_smooth(data, size):
    """
    Smooth padded pxt with Savitzky-Golay filter
    :param data: input pxt
    :param size: how many x points be padded at both ends of x
    """
    data_expanded = np.zeros((data.shape[0], data.shape[1], data.shape[2] + 2 * size))
    data_expanded[:, :, size: -size] = data
    data_expanded = signal.savgol_filter(data_expanded, sf_range, 2, axis=2)
    data_expanded = signal.savgol_filter(data_expanded, sf_range, 2, axis=1)
    smooth_data = data_expanded[:, :, size: -size]
    return smooth_data


def main():
    # === create output folder ===
    run_ = 0
    while run_ < 100:
        op_path = '{}/{}_{}_{}'.format(op_dir, ip_name.split('.')[0], run_id, run_)
        if os.path.exists(op_path):
            print(op_path, 'exist')
            run_ += 1
            pass
        elif run_ == 99:
            sys.exit('Already created 100 folders')
        else:
            os.makedirs(op_path)
            break

    # === load data ===
    with np.load('{}/{}'.format(ip_dir, ip_name)) as npz:
        x = npz['x']
        x_points = len(x)
        t = npz['t']
        true_g = npz['true_g']
        true_h = npz['true_h']
        cr = npz['central_range']
        true_pxt = npz['true_pxt']
        noisy_pxt = npz['noisy_pxt']

        print('x shape {}, t shape {}, pxt shape {}'.format(x.shape, t.shape, true_pxt.shape))

    # === record hyper-parameter ===
    with open(op_path + '/train.log', 'w') as log:
        log.write('seed: {} learning rate gh: {} p: {}\n'.format(seed, gh_learning_rate, p_learning_rate))
        log.write('patience gh: {} p: {}\n'.format(gh_patience, p_patience))
        log.write('batch size: {} t_sro: {}\n'.format(batch_size, t_sro))
        log.write('recur win gh: {} p: {}\n'.format(gh_recur_win, p_recur_win))
        log.write('smooth p: {}, sf_range: {} \n'.format(smooth_p, sf_range))
        log.write('smooth gh: {}, initial sigma: {} \n'.format(smooth_gh, smooth_gh_sigma))
        log.write('t_sro: {} p epoch factor: {}\n'.format(t_sro, p_epoch_factor))
        log.write('Initial error of pxt before smooth: {:.8f} ratio {:.8f}\n'.format(
                    np.sum((noisy_pxt - true_pxt)**2)**0.5,
                    np.sum((noisy_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))

        smooth_pxt = sf_smooth(noisy_pxt, 5)
        log.write('Initial error of pxt after smooth: {:.8f} ratio {:.8f}\n'.format(
                    np.sum((smooth_pxt - true_pxt)**2)**0.5,
                    np.sum((smooth_pxt - true_pxt)**2)**0.5/np.sum(true_pxt**2)**0.5))

    # === smooth noisy pxt with Savitzky-Golay filter ===
    if smooth_p:
        initial_pxt = np.copy(smooth_pxt)
    else:
        initial_pxt = np.copy(noisy_pxt)

    # === pxt will be updated when training p ===
    update_pxt = np.copy(initial_pxt)

    # === calculate initial gh by linear least square ===
    lsq = FPLeastSquare(x_coord=x, t_sro=t_sro)
    lsq_g, lsq_h, dt_, _ = lsq.lsq_wo_t(pxt=initial_pxt, t=t)

    if use_true_gh:
        gg_v, hh_v = true_g.reshape(-1, 1, 1), true_h.reshape(-1, 1, 1)
    else:
        gg_v, hh_v = lsq_g.reshape(-1, 1, 1), lsq_h.reshape(-1, 1, 1)

    with open(op_path + '/train.log', 'a') as log:
        log.write('Linear least square method\n')
        log.write('Error of g: {:.8f}, h: {:.8f} \t '
                  'Ratio Error of g: {:.8f}, h: {:.8f}\t'
                  'Central Error of g: {:.8f}, h: {:.8f}\n'.format(
                    np.sum((gg_v[:, 0, 0] - true_g) ** 2),
                    np.sum((hh_v[:, 0, 0] - true_h) ** 2),
                    np.sum((gg_v[:, 0, 0] - true_g) ** 2) / np.sum(true_g ** 2),
                    np.sum((hh_v[:, 0, 0] - true_h) ** 2) / np.sum(true_h ** 2),
                    np.sum((gg_v[cr[0]:cr[1], 0, 0] - true_g[cr[0]:cr[1]]) ** 2) / np.sum(true_g[cr[0]:cr[1]] ** 2),
                    np.sum((hh_v[cr[0]:cr[1], 0, 0] - true_h[cr[0]:cr[1]]) ** 2) / np.sum(true_h[cr[0]:cr[1]] ** 2)))

    # === network for gh training ===
    fpe_net = FPENet(x_coord=x, name=model_name, t_sro=t_sro)
    gh_nn = fpe_net.recur_train_gh(learning_rate=gh_learning_rate, loss=Loss.sum_square)
    # === network for p training ===
    p_nn = fpe_net.recur_train_p(learning_rate=p_learning_rate, loss=Loss.sum_square,
                                 fix_g=gg_v, fix_h=hh_v)

    # === process data to prepare samples for gh training ===
    win_x, win_t, win_y, _ = PxtData.get_recur_win_center(update_pxt, t, gh_recur_win)
    train_gh_x = np.copy(win_x)
    train_gh_y = np.copy(win_y)
    train_gh_t = np.copy(win_t)

    # === process data to prepare samples for p training ===
    win_x, win_t, win_y, win_id = PxtData.get_recur_win_center(update_pxt, t, p_recur_win)
    train_p_x = np.ones((1, x_points, 1))
    train_p_p = np.copy(win_x)
    train_p_y = np.copy(win_y)
    train_p_t = np.copy(win_t)

    # === process true pxt for control ===
    true_train_x, _, _, _ = PxtData.get_recur_win_center(true_pxt, t, p_recur_win)

    # === recording ===
    with open(op_path + '/train.log', 'a') as log:
        log.write('Initial error of p: {} \n'.format(np.sum((train_p_p - true_train_x)**2)**0.5))

    # === train gh and p in an alternating way ===
    n_sample = train_p_p.shape[0]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for iter_ in range(n_iter):

        # === smooth gh with Gaussian filter, sigma gradually decreases ===
        if smooth_gh:
            gg_v[:, 0, 0] = ndimage.gaussian_filter1d(gg_v[:, 0, 0], sigma=smooth_gh_sigma/(iter_+1))
            hh_v[:, 0, 0] = ndimage.gaussian_filter1d(hh_v[:, 0, 0], sigma=smooth_gh_sigma/(iter_+1))

        # === train gh ===
        gh_nn.get_layer(name=model_name + 'g').set_weights([gg_v])
        gh_nn.get_layer(name=model_name + 'h').set_weights([hh_v])

        es = callbacks.EarlyStopping(verbose=verb, patience=gh_patience)
        gh_nn.fit([train_gh_x, train_gh_t], train_gh_y, epochs=gh_epoch, batch_size=batch_size, verbose=verb,
                  callbacks=[es], validation_split=0.2)

        gg_v = gh_nn.get_layer(name=model_name + 'g').get_weights()[0]
        hh_v = gh_nn.get_layer(name=model_name + 'h').get_weights()[0]

        gh_nn_pred = gh_nn.predict([train_gh_x, train_gh_t])

        # === recording gh training result ===
        with open(op_path + '/train.log', 'a') as log:
            log.write('Iter: {} \n'.format(iter_))
            log.write('Prediction error after gh training: {:.8f}\n'.format(np.sum((train_gh_y - gh_nn_pred)**2)))
            log.write('Error of g: {:.8f}, h: {:.8f} \t '
                      'Ratio Error of g: {:.8f}, h: {:.8f}\t'
                      'Central Error of g: {:.8f}, h: {:.8f}\n'.format(
                        np.sum((gg_v[:, 0, 0] - true_g) ** 2),
                        np.sum((hh_v[:, 0, 0] - true_h) ** 2),
                        np.sum((gg_v[:, 0, 0] - true_g) ** 2) / np.sum(true_g ** 2),
                        np.sum((hh_v[:, 0, 0] - true_h) ** 2) / np.sum(true_h ** 2),
                        np.sum((gg_v[cr[0]:cr[1], 0, 0] - true_g[cr[0]:cr[1]]) ** 2) / np.sum(true_g[cr[0]:cr[1]] ** 2),
                        np.sum((hh_v[cr[0]:cr[1], 0, 0] - true_h[cr[0]:cr[1]]) ** 2) / np.sum(true_h[cr[0]:cr[1]] ** 2)))

        # === train p ===
        p_nn.get_layer(name=model_name + 'g').set_weights([gg_v])
        p_nn.get_layer(name=model_name + 'h').set_weights([hh_v])

        total_train_p_loss_before = 0
        total_train_p_loss_after = 0

        for i in range(n_sample):
            sample_id, t_id = win_id[i]
            print('Training P, Sample id: {}, time id {}'.format(sample_id, t_id))
            p_nn.get_layer(name=model_name + 'p').set_weights([train_p_p[i].reshape(-1, 1, 1)])
            es = callbacks.EarlyStopping(monitor='loss', verbose=verb, patience=p_patience,
                                         restore_best_weights=True)
            total_train_p_loss_before += p_nn.evaluate([train_p_x, train_p_t[i:i + 1, ...]], train_p_y[i:i + 1, ...])

            p_nn.fit([train_p_x, train_p_t[i:i + 1, ...]], train_p_y[i:i + 1, ...],
                     epochs=iter_ // p_epoch_factor + 1, verbose=verb, callbacks=[es])

            total_train_p_loss_after += p_nn.evaluate([train_p_x, train_p_t[i:i + 1, ...]], train_p_y[i:i + 1, ...])
            update_pxt[sample_id, t_id] = p_nn.get_layer(name=model_name + 'p').get_weights()[0][:, 0, 0]

        # === process data for next round training ===
        if smooth_p:
            update_pxt = sf_smooth(update_pxt, 5)

        win_x, win_t, win_y, _ = PxtData.get_recur_win_center(update_pxt, t, gh_recur_win)
        train_gh_x = np.copy(win_x)
        train_gh_y = np.copy(win_y)
        train_gh_t = np.copy(win_t)

        win_x, win_t, win_y, _ = PxtData.get_recur_win_center(update_pxt, t, p_recur_win)
        train_p_p = np.copy(win_x)

        # === recording p training result ===
        with open(op_path + '/train.log', 'a') as log:
            log.write('Total error of p training before: {:.8f}, after: {:.8f}\n'.format(
                        total_train_p_loss_before, total_train_p_loss_after))
            log.write('Error to true p: {:.8f} ratio {:.8f}, Error to noisy p: {:.8f} ratio {:.8f} \n'.format(
                        np.sum((train_gh_x - true_train_x)**2)**0.5,
                        (np.sum((train_gh_x - true_train_x) ** 2)/np.sum(true_train_x**2))**0.5,
                        np.sum((train_p_p - true_train_x) ** 2)**0.5,
                        (np.sum((train_p_p - true_train_x) ** 2)/np.sum(true_train_x**2))**0.5))

        # === save training result ===
        np.savez_compressed('{}/iter{}.npz'.format(op_path, iter_), pxt=update_pxt, g=gg_v[:, 0, 0], h=hh_v[:, 0, 0])


if __name__ == '__main__':
    main()
