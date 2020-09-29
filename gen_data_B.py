import numpy as np
import sys
import math
import matplotlib.pyplot as plt
sys.path.insert(1, './ifpeModules')
from FokkerPlankEqn import FokkerPlankForward as FPF
import B_config as config   # Local Script

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

# t_init_min = 0.005      # id 0
# t_init_max = 0.010
# y_min = 0.6
# y_max = 0.9

# t_init_min = 0.005
# t_init_max = 0.010
# y_min = 0.5
# y_max = 0.8

# t_init_min = 0.00003
# t_init_max = 0.00008
# y_min = 0.04
# y_max = 0.07
# D = 0.5

# D = 0.0015  # ID 4
# y_min = 0.06
# y_max = 0.09
# t_init_min = 0.02
# t_init_max = 0.04
# THETA = 2.86

# D = 0.0015  # ID 5
# y_min = 1.8
# y_max = 1.8
# t_init_min = 1.5
# t_init_max = 1.5
# THETA = 2.86

# t_init_min = 0.005      # ID 6 / 7
# t_init_max = 0.010
# y_min = 0.9
# y_max = 1.2
# D = 0.5
# THETA = 2.86

# t_init_min = 0.004        # ID 8
# t_init_max = 0.008
# y_min = 0.5
# y_max = 0.7

# t_init_min = 0.005          # ID 12
# t_init_max = 0.010
# y_min = 0.6
# y_max = 0.9

D = 0.5
# D = 0.00005             # ID 20
THETA = 2.86
# THETA = 0.01
# t_init_min = 0.5
# t_init_max = 0.6
# y_min = 0.5
# y_max = 0.8

t_init_min = 0.005
t_init_max = 0.010
y_min = 0.3
y_max = 0.7

# print(1.-np.exp(-2. * THETA * 60))
# print(D / THETA)
# print(D * (1.-np.exp(-2. * THETA * 0.6)) / THETA)
# sys.exit()


def p_initial(x, mu_min, mu_max, sigma_min, sigma_max, gau_no=2, seed=None):
    # x_points = len(x)
    # x_min = np.min(x)
    # x_max = np.max(x)
    p0 = np.zeros(x_points)
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
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
    # g = THETA * x
    # g = 1/x + 0.002         # ID 20
    # g = 1/x - 0.2
    # h = D * np.ones(x_points)
    # ========================== boltz
    g = x - 0.1
    # h = x ** 2 * 0.1 / 2
    h = x ** 2 / 4

    print('x:', x)
    print(g, h)
    true_pxt = np.zeros((n_sample, t_points+1, x_points))
    true_pxt[:, 0, :] = x
    noisy_pxt = np.zeros((n_sample, t_points+1, x_points))
    noisy_pxt[:, 0, :] = x

    f_true_pxt = np.zeros((n_sample, t_points+1, x_points))
    f_true_pxt[:, 0, :] = x
    f_noisy_pxt = np.zeros((n_sample, t_points+1, x_points))
    f_noisy_pxt[:, 0, :] = x
    noise0 = np.random.randn(n_sample, t_points, x_points)  # normal distribution center N(0, 1) error larger
    noise = np.random.randn(n_sample, t_points, x_points)  # normal distribution center N(0, 1) error larger
    seed_list = np.random.randint(0, 10000, n_sample)
    for i in range(n_sample):
        # Generate Data
        print('Generating sample {}'.format(i))
        mu_var = create_ou()
        p = histogram_ou(x, mu_var)
        true_pxt[i, 1:, :] = p

        # addition noise
        noisy_p = p + sigma * noise0[i, ...]
        # multiplication noise
        # noisy_p = p * (0.01 * noise[i, ...] + 1)
        noisy_p[noisy_p < 0] = 0

        print('E P0', np.sum((p[0, :] - noisy_p[0, :]) ** 2))
        print('Sum', np.sum(p[-1]) * x_gap)

        noisy_pxt[i, 1:, :] = noisy_p

        t_factor = 10
        # p0 = p_initial(x, mu_min=0.4, mu_max=0.8, sigma_min=0.05, sigma_max=0.08, gau_no=1,
        #                seed=int(100 * seed_list[i]))
        pxt = FPF.ghx2pxt(g, h, p[0], x_gap, t_gap=t_gap/t_factor, t_points=t_points*t_factor, rk=4, t_sro=7)
        # pxt = FPF.ghx2pxt(g, h, p0, x_gap, t_gap=t_gap/t_factor, t_points=t_points*t_factor, rk=4, t_sro=7)
        f_true_pxt[i, 1:, :] = pxt[0:-1:t_factor, :]

        # addition noise
        f_noisy_p = pxt[0:-1:t_factor, :] + sigma * noise[i, ...]
        # f_noisy_p = pxt[0:-1:t_factor, :] + sigma * noise[i, ...] * pxt[0:-1:t_factor, :]
        # / np.max(pxt[0:-1:t_factor, :])
        # multiplication noise
        # f_noisy_p = pxt[0:-1:t_factor, :] * (sigma * noise[i, ...] + 1)
        f_noisy_p[f_noisy_p < 0] = 0
        # print(np.sum(f_noisy_p, axis=1))

        # f_noisy_p /= np.sum(f_noisy_p, axis=1)[:, None]
        # f_noisy_p *= 100

        # print('E P0', np.sum((pxt[0:-1:t_factor, :] - f_noisy_p[0, :]) ** 2)/np.sum(pxt[0:-1:t_factor, :]**2)**0.5)
        # print('Sum', np.sum(p0))
        f_noisy_pxt[i, 1:, :] = f_noisy_p
        print(np.sum(f_noisy_pxt[i, -1, :]))
        # sys.exit()

        # plt.figure(figsize=[12, 8])
        # plt.plot(x[:100], true_pxt[i, 1, :100], 'k-', label='p_initial', linewidth=4)
        # plt.plot(x[:100], true_pxt[i, -1, :100], 'r-', label='p_final', linewidth=4)
        # # plt.plot(x, f_true_pxt[i, 1, :], 'y-', label='f_p_initial', linewidth=4)
        # plt.plot(x[:100], f_true_pxt[i, -1, :100], 'g-', label='f_p_final', linewidth=4)
        # # plt.plot(x, f_noisy_pxt[i, 1, :], 'r.', label='p_initial')
        # # plt.plot(x, f_noisy_pxt[i, -1, :], 'b^', label='p_final')
        # plt.legend(fontsize=30)
        # plt.ion()
        # plt.pause(0.6)
        # plt.close()
        # # sys.exit()
        # # plt.show()

        # plt.figure(figsize=[12, 8])
        # plt.plot(x, true_pxt[i, 1, :], 'k-', label='p_initial', linewidth=4)
        # plt.plot(x, true_pxt[i, -1, :], 'r-', label='p_final', linewidth=4)
        # plt.plot(x, f_true_pxt[i, 1, :], 'y-', label='f_p_initial')
        # plt.plot(x, f_true_pxt[i, -1, :], 'b-', label='f_p_final', linewidth=4)
        # plt.plot(x, noisy_pxt[i, -1, :], 'r.', label='p_initial')
        # plt.plot(x, f_noisy_pxt[i, -1, :], 'b^', label='p_final')
        # plt.legend()
        # plt.ion()
        # plt.pause(0.6)
        # plt.close()
        # sys.exit()
        # plt.show()


        # sys.exit()

    print(np.min(f_noisy_pxt[:, 1:, :]))
    # print(f_true_pxt[0, 0, :])
    error = f_true_pxt[:, 1:, :] - f_noisy_pxt[:, 1:, :]
    print(np.sum(error**2))
    print(np.sum(f_true_pxt[:, 1:, :]**2))
    print(np.sum(error**2)/np.sum(f_true_pxt[:, 1:, :]**2))
    print((np.sum(error ** 2) / np.sum(f_true_pxt[:, 1:, :] ** 2))**0.5)
    # np.save('./Pxt/B_OU_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma), true_pxt)
    # np.save('./Pxt/B_OU_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma), noisy_pxt)
    np.save('./Pxt/B_f_{}_pxt_{}_sigma{}.npy'.format(run_id, seed, sigma), f_true_pxt[:, :, :100])
    np.save('./Pxt/B_f_{}_noisy_{}_sigma{}.npy'.format(run_id, seed, sigma), f_noisy_pxt[:, :, :100])
    np.savez_compressed('./Pxt/test_id{}_{}_sigma{}'.format(run_id, seed, sigma), x=x[:100],
                        true_pxt=f_true_pxt[:, :, :100], noisy_pxt=f_noisy_pxt[:, :, :100])


if __name__ == '__main__':
    ou_main_run()
