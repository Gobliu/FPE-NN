import numpy as np
import matplotlib.pyplot as plt
# import config
# from PartialDerivativeGrid import PartialDerivativeGrid as PDeGrid


# x_min = config.X_MIN
# x_max = config.X_MAX
# x_points = config.X_POINTS
# x = np.linspace(x_min, x_max, num=x_points, endpoint=False)

font = {'size': 24}
plt.rc('font', **font)
plt.rc('axes', linewidth=2)

# directory = './CSV'
#
# fig_g = plt.figure(figsize=[12, 8])
# d1, d2 = mean_l2_gh.shape
# d2 = 6
# x = np.arange(0, d2, 1)
# print(x)
# for i_ in range(d1):
#     plt.errorbar(x, mean_l2_p[i_, :d2], yerr=std_l2_p[i_, :d2], fmt='o',
#                  label='{:.2e}'.format(noise_factor[i_]))
# plt.ylim(-1e6, 2.8e7)
# # plt.yscale('log')
# plt.xlabel('Delta t',  fontweight='bold')
# plt.ylabel('L2 error', fontsize=24, fontweight='bold')
# plt.title('L2 Error of Noisy P\n', fontweight='bold',
#           horizontalalignment='center', verticalalignment='baseline')
# plt.legend(loc='upper left', bbox_to_anchor=[0, 1], ncol=2, title='Noise Factor')
# plt.show()

# directory = './Result/pseudoB/{}_id{}_p{}_win{}{}'.format(9, 0, 10, 9, 9)
# directory = './Result/OU/{}_id{}_p{}_win{}{}'.format(0, 4, 10, 9, 9)
# directory = './Result/Bessel/{}_id{}_p{}_win{}{}'.format(1, 6, 10, 9, 9)
# directory = '/home/liuwei/Cluster/Bessel/id{}_{}_p{}_win{}{}'.format(2016, 2, 10, 13, 13)
# directory = '/home/liuwei/GitHub/FPE-Net-Results/Bessel/id10_11_p10_win1313'
directory = '/home/liuwei/Cluster/id4_p10_win55_0'
# directory = '/home/liuwei/Cluster/Bessel/id{}_p{}_win{}{}_{}'.format(2016, 10, 13, 13, 0)
# real_g = 1/x - 0.2
# real_h = 0.0013 * np.ones(x_points)

iter_range = 90
error_g = np.zeros(iter_range)
error_h = np.zeros(iter_range)
error_p = np.zeros(iter_range)
iter_no = np.arange(0, iter_range, 1)

data = np.load('./Pxt/Tri_id{}_{}_sigma{}.npz'.format(4, 19822012, 0.01))
x = data['x']
print(x)
x_points = x.shape[0]
print(x_points)
t = data['t']
true_pxt = data['true_pxt']
noisy_pxt = data['noisy_pxt']

real_g = 0.08 * np.sin(0.2 * x) - 0.002
# h = 4.5 * np.ones(x.shape)
real_h = 0.045 * np.ones(x.shape)

p_weight = np.load(directory + '/p_weight.npy')

ip_file = open(directory+'/train.log', 'r')
log = ip_file.readlines()
valid_list = []
test_list = []
for line in log:
    if not line.startswith('Valid'):
        pass
    else:
        line = line.strip().split()
        # print(line)
        valid_list.append(float(line[2][:-1]))
        test_list.append(float(line[5]))
# print(valid_list)
valid_list = np.asarray(valid_list)
test_list = np.asarray(test_list)
pre_g = real_g
pre_h = real_h

# print(x, p_weight)
for iter_ in range(89, 90):
    # npz = np.load(directory + '/iter{}.npz'.format(iter_))
    # cal_g = npz['g']
    # cal_h = npz['h']
    cal_g = np.load(directory + '/iter{}_gg_ng.npy'.format(iter_))
    cal_h = np.load(directory + '/iter{}_hh_ng.npy'.format(iter_))
    error_g[iter_] = np.sum((cal_g - real_g)**2)
    error_h[iter_] = np.sum((cal_h - real_h)**2)

    print(cal_h, real_h)
    print(np.sum((cal_g - real_g)**2), np.sum(real_g**2))
    print(np.sum((cal_h - real_h) ** 2), np.sum(real_h ** 2))
    print(np.sum((cal_g - real_g)**2)/np.sum(real_g**2), np.sum((cal_h - real_h)**2)/np.sum(real_h**2))
    print(np.sum(p_weight * (cal_g - real_g) ** 2) / np.sum(p_weight * real_g ** 2),
          np.sum(p_weight * (cal_h - real_h) ** 2) / np.sum(p_weight * real_h ** 2))

    plt.figure(figsize=[12, 8])
    plt.plot(x, real_g, 'k-', linewidth=4, label='Real')
    plt.plot(x, cal_g, 'ro', linewidth=4, label='Cal')
    plt.plot(x, pre_g, 'b+', linewidth=4, label='Pre')
    # plt.plot(x, old_g, 'b+', linewidth=4, label='Old Cal')
    # plt.axvline(x=0.25, ls='--', c='blue', linewidth=4)
    # plt.axvline(x=0.81, ls='--', c='blue', linewidth=4)
    plt.xlabel('x',  fontweight='bold')
    plt.ylabel('g', fontsize=24, fontweight='bold')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title('g(x) iter {}'.format(iter_), fontweight='bold',
              horizontalalignment='center', verticalalignment='baseline')
    plt.legend(loc='upper left', bbox_to_anchor=[0.4, 1], ncol=1)
    plt.show()

    plt.figure(figsize=[12, 8])
    plt.plot(x, real_h, 'k-', linewidth=4, label='Real')
    plt.plot(x, cal_h, 'ro', linewidth=4, label='Cal')
    plt.plot(x, pre_h, 'b+', linewidth=4, label='Pre')
    # plt.axvline(x=0.25, ls='--', c='blue', linewidth=4)
    # plt.axvline(x=0.81, ls='--', c='blue', linewidth=4)
    # plt.plot(x, old_h, 'b+', linewidth=4, label='Old Cal')
    plt.xlabel('x',  fontweight='bold')
    plt.ylabel('h', fontsize=24, fontweight='bold')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title('h(x) iter {}'.format(iter_), fontweight='bold',
              horizontalalignment='center', verticalalignment='baseline')
    plt.legend(loc='upper left', bbox_to_anchor=[0.4, 1], ncol=1)
    plt.show()

    pre_g = cal_g
    pre_h = cal_h

