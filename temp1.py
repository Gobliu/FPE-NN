import numpy as np
import matplotlib.pyplot as plt


font = {'size': 18}

plt.rc('font', **font)

plt.rc('axes', linewidth=2)

legend_properties = {'weight': 'bold'}

# OU
g11 = np.load('/home/liuwei/Cluster/OU/id1_p10_win1313_3/iter136_gg_ng.npy')
h11 = np.load('/home/liuwei/Cluster/OU/id1_p10_win1313_3/iter136_hh_ng.npy')
g21 = np.load('/home/liuwei/Cluster/OU/id1_p10_win1313_4/iter47_gg_ng.npy')
h21 = np.load('/home/liuwei/Cluster/OU/id1_p10_win1313_4/iter47_hh_ng.npy')
g31 = np.load('/home/liuwei/Cluster/OU/id1_p10_win1717_2/iter46_gg_ng.npy')
h31 = np.load('/home/liuwei/Cluster/OU/id1_p10_win1313_2/iter46_hh_ng.npy')

x1 = np.linspace(-0.01, 0.1, num=110, endpoint=False)
OU_real_g = 2.86 * x1
# OU_real_h = 0.0013 * np.ones(x1.shape)
OU_real_h = 0.045 * np.ones(x1.shape)

# Bessel
g12 = np.load('/home/liuwei/Cluster/Bessel/id10_p10_win1313_2/iter73_gg_ng.npy')
h12 = np.load('/home/liuwei/Cluster/Bessel/id10_p10_win1313_2/iter73_hh_ng.npy')
g22 = np.load('/home/liuwei/GitHub/Result/Bessel/id12_p10_win1717_2/iter157_gg_ng.npy')
h22 = np.load('/home/liuwei/GitHub/Result/Bessel/id12_p10_win1717_2/iter157_hh_ng.npy')
g32 = np.load('/home/liuwei/GitHub/Result/Bessel/id12_p10_win1717_0/iter152_gg_ng.npy')
h32 = np.load('/home/liuwei/GitHub/Result/Bessel/id12_p10_win1717_0/iter152_hh_ng.npy')
# g5 = np.load('/home/liuwei/Cluster/Bessel/id12_p10_win1313_0/iter109_gg_ng.npy')
# h5 = np.load('/home/liuwei/Cluster/Bessel/id12_p10_win1313_0/iter109_hh_ng.npy')

x2 = np.linspace(0.1, 1.1, num=100, endpoint=False)
B_real_g = 1 / x2 - 0.2
B_real_h = 0.5 * np.ones(x2.shape)
#
# print(np.sum((g1 - B_real_g)**2)/np.sum(B_real_g**2),
#       np.sum((h1 - B_real_h)**2)/np.sum(B_real_h**2))
#
# print(np.sum((g3 - B_real_g)**2)/np.sum(B_real_g**2),
#       np.sum((h3 - B_real_h)**2)/np.sum(B_real_h**2))

# # Boltz
g13 = np.load('/home/liuwei/Cluster/Boltz/id1_p10_win1717_0/iter440_gg_ng.npy')
h13 = np.load('/home/liuwei/Cluster/Boltz/id1_p10_win1717_0/iter440_hh_ng.npy')
g23 = np.load('/home/liuwei/Cluster/Boltz/id1_p10_win1717_2/iter129_gg_ng.npy')
h23 = np.load('/home/liuwei/Cluster/Boltz/id1_p10_win1717_2/iter129_hh_ng.npy')
g33 = np.load('/home/liuwei/GitHub/Result/Boltz/id1_p10_win1717_0/iter194_gg_ng.npy')
h33 = np.load('/home/liuwei/GitHub/Result/Boltz/id1_p10_win1717_0/iter194_hh_ng.npy')
# g5 = np.load('/home/liuwei/Cluster/Boltz/id1_p10_win1313_5/iter108_gg_ng.npy')
# h5 = np.load('/home/liuwei/Cluster/Boltz/id1_p10_win1313_5/iter108_hh_ng.npy')

x3 = np.linspace(0., 1., num=100, endpoint=False)
Boltz_real_g = x3 - 1
Boltz_real_h = 0.2 * x3 ** 2

plt.figure(figsize=[24, 18])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
ax = plt.subplot(2, 3, 1)
# plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x1[19:101], OU_real_g[19:101], 'k', linewidth=3, label='Ture')
plt.scatter(x1[19:101], g11[19:101], c='r', marker='d', s=40, label='Ep 0.01')
plt.scatter(x1[19:101], g21[19:101], c='b', marker='^', s=40, label='Ep 0.02')
plt.scatter(x1[19:101], g31[19:101], c='g', marker='o', s=40, label='Ep 0.03')
# plt.scatter(interval, y=0.035 * np.ones((len(interval))))
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.45, 0.52], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 4)
# plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x1[19:101], OU_real_h[19:101], 'k', linewidth=3, label='Ture')
plt.scatter(x1[19:101], h11[19:101], c='r', marker='d', s=40, label='Ep 0.01')
plt.scatter(x1[19:101], h21[19:101], c='b', marker='^', s=40, label='Ep 0.02')
plt.scatter(x1[19:101], h31[19:101], c='g', marker='o', s=40, label='Ep 0.03')
# plt.scatter(interval, y=0.035 * np.ones((len(interval))))
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.45, 0.85], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')
# plt.show()


plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
ax = plt.subplot(2, 3, 2)
# plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x2[12:84], B_real_g[12:84], 'k', linewidth=3, label='Ture')
plt.scatter(x2[12:84], g12[12:84], c='r', marker='d', s=40, label='Ep 0.01')
plt.scatter(x2[12:84], g22[12:84], c='b', marker='^', s=40, label='Ep 0.02')
plt.scatter(x2[12:84], g32[12:84], c='g', marker='o', s=40, label='Ep 0.03')
# plt.plot(x2, g5, linewidth=3, label='5% noise')
# plt.scatter(interval, y=0.035 * np.ones((len(interval))))
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.45, 0.85], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 5)
# plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x2[12:84], B_real_h[12:84], 'k', linewidth=3, label='Ture')
plt.scatter(x2[12:84], h12[12:84], c='r', marker='d', s=40, label='Ep 0.01')
plt.scatter(x2[12:84], h22[12:84], c='b', marker='^', s=40, label='Ep 0.02')
plt.scatter(x2[12:84], h32[12:84], c='g', marker='o', s=40, label='Ep 0.03')
# plt.plot(x2, h5, linewidth=3, label='5% noise')
# plt.scatter(interval, y=0.035 * np.ones((len(interval))))
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.10, 0.85], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')
# plt.show()

# range_3 = [14, 87]
# plt.figure(figsize=[16, 18])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.95, wspace=0.25, hspace=0.2)
ax = plt.subplot(2, 3, 3)
# plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x3[14:87], Boltz_real_g[14:87], 'k', linewidth=3, label='Ture')
plt.scatter(x3[14:87], g13[14:87], c='r', marker='d', s=40, label='Ep 0.01')
plt.scatter(x3[14:87], g23[14:87], c='b', marker='^', s=40, label='Ep 0.02')
plt.scatter(x3[14:87], g33[14:87], c='g', marker='o', s=40, label='Ep 0.03')
# plt.scatter(x1[16:110], OU_g1[16:110], c='b', marker='+', s=50, label='untrained P')
# plt.scatter(x1[16:110], OU_g2[16:110], c='r', marker='d', s=50, label='trained P')
# plt.scatter(interval, y=0.035 * np.ones((len(interval))))
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.45, 0.52], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')

ax = plt.subplot(2, 3, 6)
# plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(x3[14:87], Boltz_real_h[14:87], 'k', linewidth=3, label='Ture')
plt.scatter(x3[14:87], h13[14:87], c='r', marker='d', s=40, label='Ep 0.01')
plt.scatter(x3[14:87], h23[14:87], c='b', marker='^', s=50, label='Ep 0.02')
plt.scatter(x3[14:87], h33[14:87], c='g', marker='o', s=50, label='Ep 0.03')
# plt.scatter(interval, y=0.035 * np.ones((len(interval))))
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.10, 0.85], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')
plt.show()


