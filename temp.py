import numpy as np
import matplotlib.pyplot as plt

data_ = np.loadtxt('./stock/data_x.dat')

FTSE = data_[:, :100]
print(FTSE.shape)
FTSE_dif = np.sum((FTSE[:-1] - FTSE[1:])**2, axis=1)
print(max(FTSE_dif), min(FTSE_dif), np.std(FTSE_dif), np.mean(FTSE_dif))
FTSE_mean, FTSE_std = np.mean(FTSE_dif), np.std(FTSE_dif)

# ##################
interval = [0]
for i in range(len(FTSE_dif)):
    if FTSE_dif[i] > np.mean(FTSE_dif) + 2.5 * np.std(FTSE_dif):
        # print(i)
        interval.append(i)
interval.append(len(FTSE_dif))
print(FTSE_dif[1800:1805])
print(interval)
gap = np.zeros((len(interval)-1))
for i in range(len(interval)-1):
    gap[i] = interval[i+1]-interval[i]
print(np.sum(gap > 30))
print(gap)
# ##################

FTSE_fragment = [[0, 41], [42, 85], [94, 141], [142, 188], [189, 231], [232, 273], [286, 328], [329, 372],
                 [389, 438], [439, 488], [497, 532], [533, 572], [573, 616], [617, 639], [691, 730],
                 [731, 770], [771, 800], [801, 840], [841, 884], [908, 969], [990, 1023], [1042, 1073],
                 [1074, 1107], [1108, 1141], [1142, 1182], [1184, 1220], [1221, 1259], [1264, 1324],
                 [1402, 1441], [1442, 1481], [1482, 1523], [1524, 1563], [1564, 1600], [1601, 1640],
                 [1641, 1680], [1681, 1720], [1721, 1753], [1770, 1803], [1804, 1840], [1845, 1885],
                 [1886, 1925], [1926, 1965], [1966, 2005], [2006, 2045], [2046, 2083], [2137, 2177],
                 [2178, 2213], [2214, 2247], [2248, 2286], [2313, 2369]]

FTSE_fragment = [[0, 30], [30, 60], [94, 124], [124, 154], [154, 184], [189, 219], [219, 249], [286, 316], [316, 346],
                   [389, 419], [419, 449], [449, 479], [497, 527], [533, 563], [563, 593], [691, 721], [721, 751],
                   [751, 781], [781, 811], [811, 841], [841, 871], [908, 938], [938, 968], [990, 1020], [1042, 1072],
                   [1074, 1104], [1108, 1138], [1184, 1214], [1214, 1244], [1264, 1294],
                   [1402, 1432], [1432, 1462], [1462, 1492], [1492, 1522], [1524, 1554], [1564, 1594], [1594, 1624],
                   [1624, 1654], [1654, 1684], [1684, 1714], [1714, 1744], [1770, 1800], [1800, 1830], [1845, 1875],
                   [1875, 1905], [1905, 1935], [1935, 1965], [1965, 1995], [1995, 2025], [2025, 2055], [2137, 2167],
                   [2167, 2197], [2214, 2244], [2244, 2274], [2313, 2343]]

for f in range(len(FTSE_fragment)):
    start, end = FTSE_fragment[f][0], FTSE_fragment[f][1]
    print(end - start, start, end)
    fragment = FTSE[start:end]
    fragment_dif = np.sum((fragment[:-1] - fragment[1:])**2, axis=1)
    idx = np.arange(start, end-1)
    print(idx[fragment_dif > FTSE_mean + 2.5 * FTSE_std])



plt.figure(figsize=[24, 36])
# plt.text(-0.1, 1.10, 'A', fontsize=20, transform=ax.transAxes, fontweight='bold', va='top')
plt.plot(range(FTSE_dif.shape[0]), FTSE_dif, 'k-', linewidth=3, label='$L_{gh}$')
plt.plot(range(FTSE_dif.shape[0]), np.ones(FTSE_dif.shape[0])*np.mean(FTSE_dif), 'r-', linewidth=3, label='mean')
plt.plot(range(FTSE_dif.shape[0]), np.ones(FTSE_dif.shape[0])*(np.mean(FTSE_dif) + 2.5*np.std(FTSE_dif)), 'r--',
         linewidth=3, label='mean+2.5std')
plt.scatter(interval, y=0.035 * np.ones((len(interval))))
plt.tick_params(direction='in', width=3, length=6)
# plt.xticks(fontweight='bold')
# # plt.ylim(-0.1, 0.4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# plt.yticks(np.arange(0.005, 0.018, 0.003))
plt.legend(loc='upper left', bbox_to_anchor=[0.4, 0.92], ncol=1)
# ax.text(.5, .9, '$\mathbf{g_{error}}$', horizontalalignment='center', transform=ax.transAxes, fontsize=20)
plt.ylabel('$\mathbf{L_{gh}}$', fontweight='bold')
plt.xlabel('iter',  fontweight='bold')
# plt.show()

plt.figure()
plt.plot(FTSE[1341], 'k-', linewidth=3, label='$1341$')
plt.plot(FTSE[1342], 'b-', linewidth=3, label='$1342$')
plt.plot(FTSE[1343], 'r-', linewidth=3, label='$1343$')
plt.legend()
plt.show()

print(np.sum((FTSE[1341] - FTSE[1342])**2))
print(np.sum((FTSE[1342] - FTSE[1343])**2))
print(FTSE_dif[1340:1343])


