import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/liuwei/pyModules')
from FokkerPlankEqn import FokkerPlankForward as fpf
from PartialDerivative import PartialDerivative as PDe


def p_initial(x, mu_min, mu_max, sigma_min, sigma_max, gau_no=1, seed=2012):
    x_points = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    p0 = np.zeros(x_points)
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
    for i in range(gau_no):
        mu = np.random.random() * (mu_max - mu_min) + mu_min
        sigma = np.random.random() * (sigma_max - sigma_min) + sigma_min
        print('mu {} sigma {}'.format(mu, sigma))
        p0 += np.exp(-(x - mu)**2 / 2 * sigma ** 2)
    # p0 /= np.sum(p0)
    return p0


def main():
    runid = 0
    x_min = 0.1
    x_max = 1.1
    t_gap = 0.001
    t_points = 100
    x_points = 100
    x_gap = (x_max - x_min) / x_points
    x = np.linspace(x_min, x_max, num=x_points, endpoint=False)
    n_sample = 50
    final_pxt = np.zeros((n_sample, t_points+1, x_points))
    final_pxt[:, 0, :] = x
    for i in range(n_sample):
        p0 = p_initial(x, mu_min=0.6, mu_max=0.8, sigma_min=10, sigma_max=50, gau_no=2, seed=2012)

        g = 1/x - 0.2
        h = 0.003 * np.ones(x_points)
        pxt = fpf.ghx2pxt(g, h, p0, x_gap, t_gap=t_gap, t_points=t_points, rk=1, t_sro=7)
        final_pxt[i, 1:, ] = pxt
        plt.figure()
        plt.plot(x, pxt[0, :], 'k-', label='p_initial')
        plt.plot(x, pxt[-1, :], 'r-', label='p_final')
        # plt.plot(x, g, 'go', label='g')
        # plt.plot(x, h, 'bo', label='h')
        plt.legend
        plt.show()
    print(final_pxt.shape)
    print(final_pxt[:, 0, :])
    np.save('Pxt_ID{}_sample{}_tgap{}.npy'.format(runid, n_sample, t_gap), final_pxt)


if __name__ == '__main__':
    main()
