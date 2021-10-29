import numpy as np
import scipy as sp
import sys
import math
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from GridModules.FokkerPlankEqn import FokkerPlankForward as fpf

np.set_printoptions(suppress=True)

# ~~~~ parameter setting ~~~~
run_id = 0
seed = 19822012

x_min = -15 * math.pi
x_max = 15 * math.pi
x_points = 300          # total number of x points between x_min and x_max
x_gap = (x_max - x_min) / x_points                              # no need change
x = np.linspace(x_min, x_max, num=x_points, endpoint=False)     # no need change
slice_range = [40, -40]     # the slice range of P(x,t) to be saved in npz

t_gap = 1               # gap between time points
t_points = 50           # total number of t points for each distribution sequence
n_sample = 20          # total number of distribution sequences
xi = 0.03               # a factor for noise addition

g = 0.08 * np.sin(0.2 * x) - 0.002      # set g(x)
h = 0.045 * np.ones(x.shape)            # set h(x)

mu_range = [-15, 15]        # range of mu in gaussian distribution, for creating initial distribution
sigma_range = [5, 10]       # range of sigma in Gaussian distribution, for creating initial distribution
gau_no = 2                  # add how many Gaussian distribution in the initial distribution

op_dir = './Pxt/'           # directory of the output npz
op_name = 'test.npz'        # name of the output npz

show_plot = False           # boolean, plot or not plot the generated distributions
t_test = True               # run t_test to find region of x where P(x,t) is statistically > 0.01 * max(P(x,t)
# ~~~~ setting ends ~~~~


def p_init_gaussian():
    """ generate initial distribution by adding multiple gaussian distributions together"""

    p_init = np.zeros((n_sample, x_points))
    np.random.seed(seed)

    for sample in range(n_sample):
        p0 = np.zeros(x_points)
        for peak in range(gau_no):
            mu = np.random.random() * (mu_range[1] - mu_range[0]) + mu_range[0]
            var = np.random.random() * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
            print('sample {} peak {} mu {} sigma {}'.format(sample, peak, mu, var))
            p0 += np.exp(-(x - mu)**2 / (2 * var ** 2)) / (var * (2 * math.pi)**0.5)

        p0 /= (np.sum(p0) * x_gap)          # for normalization
        p_init[sample, :] = np.copy(p0)

    return p_init


def t_tester(x_, pxt_):
    bar = np.max(pxt_) * 0.01
    stat = np.zeros(len(x_))
    for i in range(len(x_)):
        p = pxt_[:, :, i].reshape(-1, 1)
        stat[i] = sp.stats.ttest_1samp(p, bar).statistic

    t_test_range = sp.signal.argrelextrema(abs(stat), np.less)[0]
    assert len(t_test_range) == 2, 't test range should have two entries, but got {}'.format(t_test_range)
    return t_test_range


def main():

    print('x: {} \n g: {} \n h: {}'.format(x, g, h))

    true_pxt = np.zeros((n_sample, t_points, x_points))
    t = np.zeros((n_sample, t_points, 1))
    t_factor = 10       # using t_gap/t_factor as the actual time step when calculation the evolution of P(x,t).
    noise = np.random.randn(n_sample, t_points, x_points)
    p_init = p_init_gaussian()

    for i in range(n_sample):
        # Generate Data
        print('Generating sample {}'.format(i))

        pxt = fpf.ghx2pxt(g, h, p_init[i, :], x_gap, t_gap=t_gap/t_factor, t_points=t_points*t_factor, rk=4, t_sro=7)

        pxt_idx = np.asarray(range(0, t_points*t_factor, t_factor))

        true_pxt[i, :, :] = pxt[pxt_idx]
        t[i, :, 0] = pxt_idx * (t_gap/t_factor)

    noisy_pxt = (1 + xi * noise) * true_pxt   # multiplication noise

    print(np.min(noisy_pxt))
    noisy_pxt[noisy_pxt < 0] = 0

    sliced_g = g[slice_range[0]:slice_range[1]]
    sliced_h = h[slice_range[0]:slice_range[1]]
    sliced_x = x[slice_range[0]:slice_range[1]]
    sliced_true_pxt = true_pxt[:, :, slice_range[0]:slice_range[1]]
    sliced_noisy_pxt = noisy_pxt[:, :, slice_range[0]:slice_range[1]]

    t_test_range = t_tester(sliced_x, sliced_true_pxt)

    np.savez_compressed(op_dir + op_name, t=t, true_g=sliced_g, true_h=sliced_h, t_test_range=t_test_range,
                        x=sliced_x, true_pxt=sliced_true_pxt, noisy_pxt=sliced_noisy_pxt)

    if show_plot:
        for i in range(n_sample):
            plt.figure(figsize=[9, 6])
            plt.plot(sliced_x, sliced_true_pxt[i, ...], 'k-',
                     label='p_initial', linewidth=4)
            plt.plot(sliced_x, sliced_true_pxt[i, ...], 'r-',
                     label='p_final', linewidth=4)

            plt.title('sample {}'.format(i), fontsize=20)
            plt.legend(fontsize=20)
            plt.ion()
            plt.pause(1)
            plt.close()


if __name__ == '__main__':
    main()
