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
# noise_factor = [0, 0.1**10, 0.1**9, 0.1**8, 0.1**7, 0.1**6, 0.1**5, 0.1**4, 0.1**3, 0.1**2]
#
# mean_l2_gh = np.genfromtxt(directory+'/mean_l2_error_gh.csv', delimiter=',')
# std_l2_gh = np.genfromtxt(directory+'/std_l2_error_gh.csv', delimiter=',')
# mean_l2_p = np.genfromtxt(directory+'/mean_l2_error_p.csv', delimiter=',')
# std_l2_p = np.genfromtxt(directory+'/std_l2_error_p.csv', delimiter=',')
# mean_lm_gh = np.genfromtxt(directory+'/mean_lm_error_gh.csv', delimiter=',')
# std_lm_gh = np.genfromtxt(directory+'/std_lm_error_gh.csv', delimiter=',')
# mean_lm_p = np.genfromtxt(directory+'/mean_lm_error_p.csv', delimiter=',')
# std_lm_p = np.genfromtxt(directory+'/std_lm_error_p.csv', delimiter=',')
#
# no_ps_mean_l2_gh = np.genfromtxt(directory+'/no_ps_mean_l2_error_gh.csv', delimiter=',')
# no_ps_std_l2_gh = np.genfromtxt(directory+'/no_ps_std_l2_error_gh.csv', delimiter=',')
# no_ps_mean_l2_p = np.genfromtxt(directory+'/no_ps_mean_l2_error_p.csv', delimiter=',')
# no_ps_std_l2_p = np.genfromtxt(directory+'/no_ps_std_l2_error_p.csv', delimiter=',')
# no_ps_mean_lm_gh = np.genfromtxt(directory+'/no_ps_mean_lm_error_gh.csv', delimiter=',')
# no_ps_std_lm_gh = np.genfromtxt(directory+'/no_ps_std_lm_error_gh.csv', delimiter=',')
# no_ps_mean_lm_p = np.genfromtxt(directory+'/no_ps_mean_lm_error_p.csv', delimiter=',')
# no_ps_std_lm_p = np.genfromtxt(directory+'/no_ps_std_lm_error_p.csv', delimiter=',')
#
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
# directory = './Result/Bessel/id{}_{}_p{}_win{}{}'.format(2015, 6, 10, 13, 13)
directory = './Result/Boltz/id{}_{}_p{}_win{}{}'.format(2015, 2, 10, 13, 13)
# real_g = 1/x - 0.2
# real_h = 0.0013 * np.ones(x_points)

iter_range = 350
error_g = np.zeros(iter_range)
error_h = np.zeros(iter_range)
error_p = np.zeros(iter_range)
iter_no = np.arange(0, iter_range, 1)

# load = np.load('./Pxt/pseudoB/B_OU_{}_pxt_{}_sigma{}.npy'.format(0, 19822012, 0.05))
# load = np.load('./Pxt/OU/OU_{}_pxt_{}_sigma{}.npy'.format(4, 19822012, 0.5))
# load = np.load('./Pxt/OU/4_noisy_2015_sigma0.5.npy')
# load = np.load('./Pxt/Bessel/B_f_{}_pxt_{}_sigma{}.npy'.format(2015, 19822012, 0.015))
load = np.load('./Pxt/Bessel/B_f_{}_pxt_{}_sigma{}.npy'.format(2015, 19822012, 0.015))
x = load[0, 0, :]
# real_g = 1/x - 0.2
# real_g = 2.86 * x
# real_h = 0.5 * np.ones(len(x))
# real_h = 0.0013 * np.ones(len(x))
# true_pxt = load[:, 1:, :]
# real_p = true_pxt[:, 0: 45, :]
# print(true_pxt.shape)
p_weight = np.load(directory + '/p_weight.npy')
# p_weight_OU = np.load(directory + '/p_weight_OU.npy')

# load = np.load('./Pxt/Boltz_{}.npz'.format(2))
# x = load['x']
# t = load['t']
# # print(t)
# # t_gap = t[0, 1] - t[0, 0]
# # print(t_gap)
# true_pxt = load['true_pxt']
# noisy_pxt = load['noisy_pxt']

# true_pxt[true_pxt < 0] = 0
# noisy_pxt[noisy_pxt < 0] = 0

real_g = x - 0.1
# real_h = x ** 2 * 0.1 / 2
# real_h = 0.5 * np.ones(x.shape)
real_h = x ** 2 / 4

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

print(x, p_weight)
for iter_ in range(200, iter_range):
    # iter_ = 140
    cal_g = np.load(directory + '/gg_iter{}_smooth.npy'.format(iter_))
    # print(cal_g.shape)
    error_g[iter_] = np.sum((cal_g - real_g)**2)
    cal_h = np.load(directory + '/hh_iter{}_smooth.npy'.format(iter_))
    error_h[iter_] = np.sum((cal_h - real_h) ** 2)

    print(np.sum((cal_g - real_g)**2), np.sum(real_g**2))
    print(np.sum((cal_h - real_h) ** 2), np.sum(real_h ** 2))
    print(np.sum((cal_g - real_g)**2)/np.sum(real_g**2), np.sum((cal_h - real_h)**2)/np.sum(real_h**2))
    print(np.sum(p_weight * (cal_g - real_g) ** 2) / np.sum(p_weight * real_g ** 2),
          np.sum(p_weight * (cal_h - real_h) ** 2) / np.sum(p_weight * real_h ** 2))

    # old_g = np.load('/home/liuwei/DRN-FPE/FPE-2019/inverse_problem/Jan2020/Results/Feb2_id0e100p1_smoothTrue_rewin9'
    #                 '/gg_iter490_smooth.npy')
    # old_h = np.load('/home/liuwei/DRN-FPE/FPE-2019/inverse_problem/Jan2020/Results/Feb2_id0e100p1_smoothTrue_rewin9'
    #                 '/hh_iter490_smooth.npy')

    # cal_p = np.load(directory + '/train_data_iter{}.npy'.format(iter_))
    # error_p[iter_] = np.sum((cal_p - real_p)**2)
    # print(iter_, np.sum((cal_p - real_p)**2))
    # print(cal_p.shape)

    plt.figure(figsize=[12, 8])
    plt.plot(x, real_g, 'k-', linewidth=4, label='Real')
    plt.plot(x, cal_g, 'ro', linewidth=4, label='Cal')
    # plt.plot(x, pre_g, 'b+', linewidth=4, label='Pre')
    # plt.plot(x, old_g, 'b+', linewidth=4, label='Old Cal')
    plt.axvline(x=0.19, ls='--', c='blue', linewidth=4)
    plt.axvline(x=0.82, ls='--', c='blue', linewidth=4)
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
    # plt.plot(x, pre_h, 'b+', linewidth=4, label='Pre')
    plt.axvline(x=0.19, ls='--', c='blue', linewidth=4)
    plt.axvline(x=0.82, ls='--', c='blue', linewidth=4)
    # plt.plot(x, old_h, 'b+', linewidth=4, label='Old Cal')
    plt.xlabel('x',  fontweight='bold')
    plt.ylabel('h', fontsize=24, fontweight='bold')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title('h(x) iter {}'.format(iter_), fontweight='bold',
              horizontalalignment='center', verticalalignment='baseline')
    plt.legend(loc='upper left', bbox_to_anchor=[0.4, 1], ncol=1)
    plt.show()

    plt.figure(figsize=[12, 8])
    # # px = np.linspace(-0.01, 0.1, num=110, endpoint=False)
    plt.plot(x, p_weight, 'k-', linewidth=4, label='Real')
    plt.axhline(y=np.max(p_weight)*0.1, ls='--', c='blue', linewidth=4)
    plt.axvline(x=0.19, ls='--', c='blue', linewidth=4)
    plt.axvline(x=0.82, ls='--', c='blue', linewidth=4)
    # # plt.plot(px, p_weight_OU, 'k-', linewidth=4, label='Cal')
    # # plt.plot(x, pre_h, 'b+', linewidth=4, label='Pre')
    # # plt.plot(x, old_h, 'b+', linewidth=4, label='Old Cal')
    # plt.xlabel('x',  fontweight='bold')
    # plt.ylabel('P', fontsize=24, fontweight='bold')
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.title('P(x)', fontweight='bold',
    #           horizontalalignment='center', verticalalignment='baseline')
    # plt.legend(loc='upper left', bbox_to_anchor=[0.1, 1], ncol=1)
    plt.show()

    pre_g = cal_g
    pre_h = cal_h
    # plt.figure(figsize=[12, 8])
    # plt.plot(x, real_p[0, 0, :], 'k-', linewidth=4, label='Real')
    # plt.plot(x, cal_p[0, 0, :], 'ro', linewidth=4, label='New Cal')
    # # plt.plot(x, old_h, 'b+', linewidth=4, label='Old Cal')
    # plt.xlabel('x',  fontweight='bold')
    # plt.ylabel('p', fontsize=24, fontweight='bold')
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.title('p(x)\n', fontweight='bold',
    #           horizontalalignment='center', verticalalignment='baseline')
    # plt.legend(loc='upper left', bbox_to_anchor=[0.4, 1], ncol=1)
    # plt.show()

# plt.figure(figsize=[12, 8])
# plt.plot(iter_no, error_g, linewidth=4)
# # plt.plot(iter_no, error_h)
# plt.xlabel('Iteration',  fontweight='bold')
# plt.ylabel('L2 error', fontsize=24, fontweight='bold')
# plt.title('L2 Error of G', fontweight='bold',
#           horizontalalignment='center', verticalalignment='baseline')
# # plt.legend(loc='upper left', bbox_to_anchor=[0.7, 1], ncol=2)
# plt.show()
# #
# plt.figure(figsize=[12, 8])
# plt.plot(iter_no, error_h, linewidth=4)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.xlabel('Iteration',  fontweight='bold')
# plt.ylabel('L2 error', fontsize=24, fontweight='bold')
# plt.title('L2 Error of H', fontweight='bold',
#           horizontalalignment='center', verticalalignment='baseline')
# # plt.legend(loc='upper left', bbox_to_anchor=[0.7, 1], ncol=2)
# plt.show()
#
# p1 = 137610 * np.ones(iter_range)
# p2 = 15088 * np.ones(iter_range)
# plt.figure(figsize=[12, 8])
# plt.plot(iter_no, error_p, linewidth=4, label='Trained P')
# plt.plot(iter_no, p1, '--', linewidth=4, label='Noisy P')
# plt.plot(iter_no, p2, '-.', linewidth=4, label='Smoothed P')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.xlabel('Iteration',  fontweight='bold')
# plt.ylabel('L2 error', fontsize=24, fontweight='bold')
# plt.title('L2 Error of P', fontweight='bold',
#           horizontalalignment='center', verticalalignment='baseline')
# plt.legend(loc='upper left', bbox_to_anchor=[0, 0.7], ncol=1)
# plt.show()

# p1 = 6905 * np.ones(iter_range)
# p2 = 6977 * np.ones(iter_range)
# plt.figure(figsize=[12, 8])
# plt.plot(iter_no, valid_list[:iter_range], 'b-', linewidth=4, label='Validation Loss')
# plt.plot(iter_no, test_list[:iter_range], 'r-', linewidth=4, label='Test Loss')
# plt.plot(iter_no, p1, '--', linewidth=4, label='Validation Reference')
# plt.plot(iter_no, p2, '-.', linewidth=4, label='Test Reference')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.xlabel('Iteration',  fontweight='bold')
# plt.ylabel('L2 error', fontsize=24, fontweight='bold')
# plt.title('L2 Error of P', fontweight='bold',
#           horizontalalignment='center', verticalalignment='baseline')
# plt.legend(loc='upper left', bbox_to_anchor=[0, 1], ncol=1)
# plt.show()
