import numpy as np
import sys
import math
import matplotlib.pyplot as plt
sys.path.insert(1, './GridModules')
from FokkerPlankEqn import FokkerPlankForward as FPF
import Boltz_config as config   # Local Script

np.set_printoptions(suppress=True)

run_id = config.RUN_ID
seed = config.SEED
x_min = config.X_MIN
x_max = config.X_MAX
x_points = config.X_POINTS
x_gap = (x_max - x_min) / x_points
t_gap = config.T_GAP
t_points = config.T_POINTS
n_sample = config.N_SAMPLE
sigma = config.SIGMA

np.random.seed(seed)

D = 0.5
THETA = 2.86

t_init_min = 0.005
t_init_max = 0.010
y_min = 0.3
y_max = 0.7


def p_initial(x, mu_min, mu_max, sigma_min, sigma_max, gau_no=2, seed=None):

    p0 = np.zeros(x_points)
    # x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
    # print(x_gap)
    if seed is not None:
        np.random.seed(seed)
    for i in range(gau_no):
        mu = np.random.random() * (mu_max - mu_min) + mu_min
        var = np.random.random() * (sigma_max - sigma_min) + sigma_min
        print('mu {} sigma {}'.format(mu, var))
        p0 += np.exp(-(x - mu)**2 / (2 * var ** 2)) / (var * (2 * math.pi)**0.5)
    p0 /= np.sum(p0)
    p0 /= x[1] - x[0]
    # print(np.sum(p0))
    return p0


def ou_mu(y, t):
    return y * np.exp(-THETA * t)


def ou_var(t):     # sigma square
    return D * (1.-np.exp(-2. * THETA * t)) / THETA


def create_ou():
    t = np.random.random()*(t_init_max - t_init_min) + t_init_min
    y = np.random.random()*(y_max-y_min) + y_min
    print('t: {}  Y: {}'.format(t, y))
    mu_var = np.zeros((t_points, 2))
    for i in range(t_points):
        mu_var[i, 0] = ou_mu(y, t)
        mu_var[i, 1] = ou_var(t)
        t += t_gap
    return mu_var


def histogram_ou(x, mu_var):
    p = np.zeros((t_points, x_points))
    for i in range(t_points):
        p[i] = np.exp(-(x-mu_var[i, 0])**2/(2*mu_var[i, 1])) / np.sqrt(2 * np.pi * mu_var[i, 1])
    return p


def ou_main_run():
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
    # ========================== boltz
    # g = x - 0.1
    # h = x ** 2 * 0.1 / 2
    # h = x ** 2 / 4
    # ========================== 2017
    g = x - 1
    h = 0.2 * x ** 2

    print('x:', x)
    print(g, h)
    true_pxt = np.zeros((n_sample, t_points, x_points))
    noisy_pxt = np.zeros((n_sample, t_points, x_points))

    f_true_pxt = np.zeros((n_sample, t_points, x_points))
    f_noisy_pxt = np.zeros((n_sample, t_points, x_points))
    t = np.zeros((n_sample, t_points, 1))

    t_factor = 10

    noise0 = np.random.randn(n_sample, t_points, x_points)  # normal distribution center N(0, 1) error larger
    noise = np.random.randn(n_sample, t_points, x_points)  # normal distribution center N(0, 1) error larger
    idx_noise = np.random.randint(-int(0.3 * t_factor), int(0.4 * t_factor), size=(n_sample, t_points))
    seed_list = np.random.randint(0, 10000, n_sample)
    for i in range(n_sample):
        # Generate Data
        print('Generating sample {}'.format(i))
        mu_var = create_ou()
        p = histogram_ou(x, mu_var)
        true_pxt[i, :, :] = p

        # addition noise
        noisy_p = p + sigma * noise0[i, ...]
        # multiplication noise
        # noisy_p = p * (0.01 * noise[i, ...] + 1)
        noisy_p[noisy_p < 0] = 0

        print('E P0', np.sum((p[0, :] - noisy_p[0, :]) ** 2))
        print('Sum', np.sum(p[-1]) * x_gap)

        noisy_pxt[i, :, :] = noisy_p

        pxt = FPF.ghx2pxt(g, h, p[0], x_gap, t_gap=t_gap/t_factor, t_points=t_points*t_factor, rk=4, t_sro=7)

        # idx_noise = np.random.randint(-int(0.3*t_factor), int(0.4*t_factor), size=t_points)
        # print(idx_noise)
        # pxt_idx = np.asarray(range(0, t_points*t_factor, t_factor)) + idx_noise[i]        # id 2015 or 2017
        pxt_idx = np.asarray(range(0, t_points*t_factor, t_factor))                         # id 2016 or 2018
        pxt_idx[pxt_idx < 0] = 0
        # print(pxt_idx)
        pxt_idx.sort()
        print('max dis:', np.max(abs(pxt_idx[1:] - pxt_idx[:-1])))

        t[i, :, 0] = pxt_idx * (t_gap/t_factor)
        # print(t[i])

        f_true_pxt[i, :, :] = pxt[pxt_idx]

        # addition noise
        f_noisy_p = pxt[pxt_idx] + sigma * noise[i, ...]

        f_noisy_p[f_noisy_p < 0] = 0
        #
        f_noisy_pxt[i, :, :] = f_noisy_p
        print(np.sum(f_noisy_pxt[i, -1, :]))

        plt.figure(figsize=[12, 8])
        plt.plot(x[:], true_pxt[i, 1, :], 'k-', label='p_initial', linewidth=4)
        plt.plot(x[:], true_pxt[i, -1, :], 'r-', label='p_final', linewidth=4)
        # plt.plot(x, f_true_pxt[i, 1, :], 'y-', label='f_p_initial', linewidth=4)
        plt.plot(x[:], f_true_pxt[i, -1, :], 'g-', label='f_p_final', linewidth=4)
        # plt.plot(x, f_noisy_pxt[i, 1, :], 'r.', label='p_initial')
        # plt.plot(x, f_noisy_pxt[i, -1, :], 'b^', label='p_final')
        plt.legend(fontsize=30)
        plt.ion()
        plt.pause(0.6)
        plt.close()
        # sys.exit()
        # plt.show()

    print(np.min(f_noisy_pxt[:, :, :100]))
    # print(f_true_pxt[0, 0, :])
    error = f_true_pxt[:, :, :100] - f_noisy_pxt[:, :, :100]
    print(np.sum(error**2))
    print(np.sum(f_true_pxt[:, :, :100]**2))
    print(np.sum(error**2)/np.sum(f_true_pxt[:, :, :100]**2))
    print((np.sum(error ** 2) / np.sum(f_true_pxt[:, :, :100] ** 2))**0.5)
    # np.save('./Pxt/B_OU_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma), true_pxt)
    # np.save('./Pxt/B_OU_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma), noisy_pxt)
    # np.save('./Pxt/B_f_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma), f_true_pxt[:, :, :100])
    # np.save('./Pxt/B_f_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma), f_noisy_pxt[:, :, :100])
    np.savez_compressed('./Pxt/Boltz_id{}_{}_sigma{}'.format(run_id, seed, sigma), x=x[:100], t=t,
                        true_pxt=f_true_pxt[:, :, :100], noisy_pxt=f_noisy_pxt[:, :, :100])


if __name__ == '__main__':
    ou_main_run()
