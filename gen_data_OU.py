import numpy as np
import sys
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

y_min = 0.04   # ID 11
y_max = 0.08
t_init_min = 0.03
t_init_max = 0.05
D = 0.0013
THETA = 2.86


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
    # p_dx = np.zeros((t_points, x_points))
    # p_dxx = np.zeros((t_points, x_points))

    for i in range(t_points):
        p[i] = np.exp(-(x-mu_var[i, 0])**2/(2*mu_var[i, 1])) / np.sqrt(2 * np.pi * mu_var[i, 1])
        # p_dx[i] = - (x-mu_var[i, 0]) * p[i] / mu_var[i, 1]
        # p_dxx[i] = ((x-mu_var[i, 0])**2 - mu_var[i, 1]) * p[i] / mu_var[i, 1]**2
    # return p, p_dx, p_dxx
    return p


def p_initial(x, mu_min, mu_max, sigma_min, sigma_max, gau_no=2, seed=None):
    p0 = np.zeros(x_points)
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
    print(x_gap)
    if seed is not None:
        np.random.seed(seed)
    for i in range(gau_no):
        mu = np.random.random() * (mu_max - mu_min) + mu_min
        var = np.random.random() * (sigma_max - sigma_min) + sigma_min
        print('mu {} sigma {}'.format(mu, var))
        p0 += np.exp(-(x - mu)**2 / (2 * var ** 2))
    p0 /= np.sum(p0)
    p0 *= x_points
    # print(np.sum(p0))
    return p0


def ou_main_run():
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
    g = THETA * x
    h = D * np.ones(x_points)
    print('x:', x)
    true_pxt = np.zeros((n_sample, t_points, x_points))
    true_pxt[:, 0, :] = x
    noisy_pxt = np.zeros((n_sample, t_points, x_points))
    noisy_pxt[:, 0, :] = x
    # total_p_dx = np.zeros((n_sample, t_points, x_points))
    # total_p_dxx = np.zeros((n_sample, t_points, x_points))

    f_true_pxt = np.zeros((n_sample, t_points, x_points))
    f_true_pxt[:, 0, :] = x
    f_noisy_pxt = np.zeros((n_sample, t_points, x_points))
    f_noisy_pxt[:, 0, :] = x
    np.random.seed(seed+1)
    noise = np.random.randn(n_sample, t_points, x_points)  # normal distribution center N(0, 1) error larger
    for i in range(n_sample):
        # Generate Data
        print('Generating sample {}'.format(i))
        mu_var = create_ou()
        p = histogram_ou(x, mu_var)
        true_pxt[i, :, :] = p

        noisy_p = p + sigma * noise[i, ...]
        print('E P0', np.sum((p[0, :] - noisy_p[0, :]) ** 2))
        print('Sum', np.sum(p[-1]) * x_gap)

        noisy_pxt[i, :, :] = noisy_p
        # total_p_dx[i] = p_dx
        # total_p_dxx[i] = p_dxx
        print(np.sum(true_pxt[i, -1, :]))

        # t_factor = 10
        # # p0 = p_initial(x, mu_min=0.5, mu_max=0.7, sigma_min=0.06, sigma_max=0.1, gau_no=2,
        # #                seed=int(100 * seed_list[i]))
        # pxt = FPF.ghx2pxt(g, h, p[0], x_gap, t_gap=t_gap/t_factor, t_points=t_points*t_factor, rk=4, t_sro=7)
        # f_true_pxt[i, 1:, :] = pxt[0:-1:t_factor, :]
        #
        # f_noisy_p = pxt[0:-1:t_factor, :] + sigma * noise[i, ...]
        # print('E P0', np.sum((p[0, :] - noisy_p[0, :]) ** 2))
        # print('Sum', np.sum(p[-1]) * x_gap)
        # f_noisy_pxt[i, 1:, :] = f_noisy_p

        # plt.figure()
        # plt.plot(x, true_pxt[i, 1, :], 'k-', label='p_initial')
        # plt.plot(x, true_pxt[i, -1, :], 'r-', label='p_final')
        # # plt.plot(x, f_true_pxt[i, 1, :], 'y-', label='p_initial')
        # # plt.plot(x, f_true_pxt[i, -1, :], 'b-', label='p_final')
        # # plt.plot(x, f_noisy_pxt[i, 1, :], 'y*', label='p_initial')
        # # plt.plot(x, f_noisy_pxt[i, -1, :], 'b^', label='p_final')
        # plt.legend()
        # plt.ion()
        # plt.pause(0.6)
        # plt.close()

    print(np.min(noisy_pxt[:, :, :110]))
    # print(f_true_pxt[0, 0, :])
    error = true_pxt[:, :, :110] - noisy_pxt[:, :, :110]
    print(np.sum(error**2))
    print(np.sum(true_pxt[:, :, :110]**2))
    print(np.sum(error**2)/np.sum(true_pxt[:, :, :110]**2))
    print((np.sum(error ** 2) / np.sum(true_pxt[:, :, :110] ** 2))**0.5)
    np.savez_compressed('./Pxt/OU_id{}_{}_sigma{}'.format(run_id, seed, sigma), x=x,
                        true_pxt=true_pxt, noisy_pxt=noisy_pxt)


if __name__ == '__main__':
    ou_main_run()
