import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
sys.path.insert(1, './GridModules')
from FokkerPlankEqn import FokkerPlankForward as FPF
import OU_config as config  # Local Script

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

# y_min = 0.04  # the rest
# y_max = 0.08
y_min = 0.05    # id 3
y_max = 0.07
t_init_min = 0.03
t_init_max = 0.05
D = 0.0013
THETA = 2.86


def ou_mu(y, t):
    return y * np.exp(-THETA * t)


def ou_var(t):     # sigma square
    return D * (1.-np.exp(-2. * THETA * t)) / THETA


def create_ou(n_point, gap):
    t_ = np.random.random()*(t_init_max - t_init_min) + t_init_min
    y = np.random.random()*(y_max-y_min) + y_min
    print('t: {}  Y: {}'.format(t_, y))
    mu_var = np.zeros((n_point, 3))
    t_np = np.zeros((n_point, 1))
    for i in range(n_point):
        mu_var[i, 0] = ou_mu(y, t_)
        mu_var[i, 1] = ou_var(t_)
        t_np[i] = copy.copy(t_)
        t_ += gap
    return mu_var, t_np


def histogram_ou(x, mu_var):
    p = np.zeros((mu_var.shape[0], x_points))

    for i in range(mu_var.shape[0]):
        p[i] = np.exp(-(x-mu_var[i, 0])**2/(2*mu_var[i, 1])) / np.sqrt(2 * np.pi * mu_var[i, 1])
    return p


def ou_main_run():
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)

    print('x:', x)
    true_pxt = np.zeros((n_sample, t_points, x_points))
    true_pxt[:, 0, :] = x
    noisy_pxt = np.zeros((n_sample, t_points, x_points))
    noisy_pxt[:, 0, :] = x
    t = np.zeros((n_sample, t_points, 1))

    np.random.seed(seed)
    noise = np.random.randn(n_sample, t_points, x_points)  # normal distribution center N(0, 1) error larger
    t_factor = 10
    np.random.seed(seed)
    idx_noise = np.random.randint(-int(0.3 * t_factor), int(0.4 * t_factor), size=(n_sample, t_points))

    for i in range(n_sample):
        # Generate Data
        print('Generating sample {}'.format(i))
        mu_var, t_one_sample = create_ou(t_points * t_factor, t_gap / t_factor)
        p = histogram_ou(x, mu_var)
        pxt_idx = np.asarray(range(0, t_points*t_factor, t_factor)) + idx_noise[i]        # odd id
        pxt_idx[pxt_idx < 0] = 0
        pxt_idx.sort()
        print('max dis:', np.max(abs(pxt_idx[1:] - pxt_idx[:-1])))
        print(p.shape, p[pxt_idx].shape)
        true_pxt[i, :, :] = p[pxt_idx]
        t[i, :] = t_one_sample[pxt_idx]
        # print('E P0', np.sum((p[0, :] - noisy_pxt[i, :, :]) ** 2))
        # print('Sum', np.sum(p[-1]) * x_gap)

        # total_p_dx[i] = p_dx
        # total_p_dxx[i] = p_dxx
        print(np.sum(true_pxt[i, -1, :]))

    # addition noise
    # noisy_pxt = true_pxt + sigma * noise        # 2015+
    noisy_pxt = true_pxt + sigma * noise * true_pxt
    noisy_pxt[noisy_pxt < 0] = 0

    for i in range(n_sample):
        plt.figure()
        plt.plot(x, true_pxt[i, 1, :], 'k-', label='p_initial')
        plt.plot(x, true_pxt[i, -1, :], 'r-', label='p_final')
        plt.plot(x, noisy_pxt[i, -1, :], 'b^', label='p_final')
        plt.legend()
        plt.ion()
        plt.pause(0.6)
        plt.close()

    print(np.min(noisy_pxt[:, :, :110]))
    # print(f_true_pxt[0, 0, :])
    error = true_pxt[:, :, :110] - noisy_pxt[:, :, :110]
    print(np.sum(error**2))
    print(np.sum(true_pxt[:, :, :110]**2))
    print(np.sum(error**2)/np.sum(true_pxt[:, :, :110]**2))
    print((np.sum(error ** 2) / np.sum(true_pxt[:, :, :110] ** 2))**0.5)
    np.savez_compressed('./Pxt/OU_id{}_{}_sigma{}'.format(run_id, seed, sigma), x=x, t=t,
                        true_pxt=true_pxt, noisy_pxt=noisy_pxt)


if __name__ == '__main__':
    ou_main_run()
