import numpy as np

from scipy import stats


# data = np.load('./Pxt/Boltz_id{}_{}_sigma{}.npz'.format(2018, 19822012, 0.02))
# data = np.load('./Pxt/Bessel_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
data = np.load('./Pxt/OU_id{}_sigma{}.npz'.format(2015, 0.5))
x = data['x']
true_pxt = data['true_pxt']
noisy_pxt = data['noisy_pxt']
print(x.shape, noisy_pxt.shape)
t_list = []

bar = np.max(noisy_pxt) * 0.01
for pos in range(x.shape[0]):
    p = noisy_pxt[:, :, pos].reshape(-1, 1)
    print(pos, stats.ttest_1samp(p, bar))
    # p *= 100
    # print(np.min(p), np.max(p))
    # mu = np.mean(p)
    # sigma = np.std(p)
    # t = (mu - bar) * (p.shape[0])**0.5 / sigma
    # t_list.append(t)
    # t__, sig = t_1samp(p, bar)
    # print(mu, sigma, (p.shape[0])**0.5, t, t__, sig, p.shape)

# print(t_list)
print(np.max(noisy_pxt))

