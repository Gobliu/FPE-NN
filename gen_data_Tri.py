import numpy as np
import sys
import math
import matplotlib.pyplot as plt
sys.path.insert(1, './GridModules')
from FokkerPlankEqn import FokkerPlankForward as FPF
import Tri_config as config   # Local Script

np.set_printoptions(suppress=True)

run_id = config.RUN_ID
seed = config.SEED
x_min = config.X_MIN * math.pi
x_max = config.X_MAX * math.pi
x_points = config.X_POINTS
x_gap = (x_max - x_min) / x_points
t_gap = config.T_GAP
t_points = config.T_POINTS
n_sample = config.N_SAMPLE
sigma = config.SIGMA

np.random.seed(seed)

D = 40
THETA = 2

t_init_min = 0.1
t_init_max = 0.2          # 2016 & 2020: 0.013
y_min = - 6 * math.pi       # 2016: 0.4 2020:0.6 others:0.5
y_max = 6 * math.pi         # 2020: 0.9 others: 0.8


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
    t_ = np.random.random()*(t_init_max - t_init_min) + t_init_min
    y = np.random.random()*(y_max-y_min) + y_min
    print('t: {}  Y: {}'.format(t_, y))
    mu_var = np.zeros((t_points, 2))
    for i in range(t_points):
        mu_var[i, 0] = ou_mu(y, t_)
        mu_var[i, 1] = ou_var(t_)
        t_ += t_gap
    return mu_var


def histogram_ou(x, mu_var):
    p = np.zeros((t_points, x_points))
    for i in range(t_points):
        p[i] = np.exp(-(x-mu_var[i, 0])**2/(2*mu_var[i, 1])) / np.sqrt(2 * np.pi * mu_var[i, 1])
    return p


def ou_main_run():
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
    # ========================== Tri
    g = - 0.4 * np.cos(0.2 * x) - 0.002 * x
    h = 4.5 * np.ones(x.shape)

    # print(x, g, h)

    # plt.figure()
    # plt.plot(g[50:-50])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(h[50:-50])
    # plt.show()
    # sys.exit()

    print('x:', x)
    print(g, h)
    true_pxt = np.zeros((n_sample, t_points, x_points))
    f_true_pxt = np.zeros((n_sample, t_points, x_points))

    t = np.zeros((n_sample, t_points, 1))

    t_factor = 10

    noise = np.random.randn(n_sample, t_points, x_points)  # normal distribution center N(0, 1) error larger
    for i in range(n_sample):
        # Generate Data
        print('Generating sample {}'.format(i))
        mu_var = create_ou()
        p = histogram_ou(x, mu_var)
        true_pxt[i, :, :] = p

        pxt = FPF.ghx2pxt(g, h, p[0], x_gap, t_gap=t_gap/t_factor, t_points=t_points*t_factor, rk=4, t_sro=7)

        pxt_idx = np.asarray(range(0, t_points*t_factor, t_factor))                         # even id

        pxt_idx.sort()

        t[i, :, 0] = pxt_idx * (t_gap/t_factor)

        f_true_pxt[i, :, :] = pxt[pxt_idx]

    # addition noise
    # f_noisy_pxt = f_true_pxt + sigma * noise[i, ...]        # 2015+
    # multiplication noise
    f_noisy_pxt = f_true_pxt + sigma * noise * f_true_pxt

    print(np.min(f_noisy_pxt))
    f_noisy_pxt[f_noisy_pxt < 0] = 0

    range_ = [40, -40]
    # range_ = [None, None]

    for i in range(n_sample):
        plt.figure(figsize=[12, 8])
        # plt.plot(x[:], true_pxt[i, 1, :], 'k-', label='p_initial', linewidth=4)
        # plt.plot(x[:], true_pxt[i, -1, :], 'r-', label='p_final', linewidth=4)
        # plt.plot(x[:], f_true_pxt[i, -1, :], 'g-', label='f_p_final', linewidth=4)
        plt.plot(x[range_[0]:range_[1]], f_true_pxt[i, 0, range_[0]:range_[1]], 'k-', label='p_initial', linewidth=4)
        plt.plot(x[range_[0]:range_[1]], true_pxt[i, -1, range_[0]:range_[1]], 'r-', label='p_final', linewidth=4)
        plt.plot(x[range_[0]:range_[1]], f_true_pxt[i, -1, range_[0]:range_[1]], 'g+', label='f_p_final', linewidth=4)

        print(np.sum(f_true_pxt[i, -1, range_[0]:range_[1]]))

        plt.legend(fontsize=30)
        plt.ion()
        plt.pause(1)
        plt.close()
        # sys.exit()
        # plt.show()

    print(np.min(f_noisy_pxt[:, :, range_[0]:range_[1]]))
    # print(f_true_pxt[0, 0, :])
    error = f_true_pxt[:, :, range_[0]:range_[1]] - f_noisy_pxt[:, :, range_[0]:range_[1]]
    print(np.sum(error**2))
    print(np.sum(f_true_pxt[:, :, range_[0]:range_[1]]**2))
    print(np.sum(error**2)/np.sum(f_true_pxt[:, :, range_[0]:range_[1]]**2))
    print((np.sum(error ** 2) / np.sum(f_true_pxt[:, :, range_[0]:range_[1]] ** 2))**0.5)
    np.savez_compressed('./Pxt/Tri_id{}_{}_sigma{}'.format(run_id, seed, sigma), x=x[range_[0]:range_[1]], t=t,
                        true_pxt=f_true_pxt[:, :, range_[0]:range_[1]],
                        noisy_pxt=f_noisy_pxt[:, :, range_[0]:range_[1]])
    print(x[range_[0]:range_[1]])


if __name__ == '__main__':
    ou_main_run()
