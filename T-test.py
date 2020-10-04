import numpy as np

from scipy import stats


# np.random.seed(7654567)  # fix seed to get the same result
# rvs = stats.norm.rvs(loc=5, scale=10, size=5000)
# print(np.mean(rvs), np.std(rvs))
# print(stats.ttest_1samp(rvs, 5.0))

# def t_1samp(list_c, u):
#     lst = list_c.copy()
#     n = len(lst)
#     s = np.std(lst)*(n**0.5)/(n-1)**0.5
#     t_ = (np.mean(lst)-u)/(s/n**0.5)
#     sig = 2*stats.t.sf(abs(t_), n-1)
#     # dic_res = [{'t值':t,'自由度':n-1,'Sig.':sig,'平均值差值':np.mean(lst)-u}]
#     # df_res = pd.DataFrame(dic_res,columns=['t值','自由度','Sig.','平均值差值'])
#     return t_, sig
#
#
data = np.load('./Pxt/Boltz_id{}_{}_sigma{}.npz'.format(2018, 19822012, 0.02))
# data = np.load('./Pxt/Bessel_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
# data = np.load('./Pxt/OU_id{}_{}_sigma{}.npz'.format(run_id, seed, sigma))
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

