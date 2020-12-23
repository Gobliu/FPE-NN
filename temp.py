import numpy as np
import matplotlib.pyplot as plt

data_ = np.loadtxt('./stock/data_x.dat')

FTSE = data_[:, 100:200]
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
print(interval)
gap = np.zeros((len(interval)-1))
for i in range(len(interval)-1):
    gap[i] = interval[i+1]-interval[i]
print(np.sum(gap > 30))
print(gap)
# ##################

# FTSE_fragment = [[0, 41], [42, 85], [94, 141], [142, 188], [189, 231], [232, 273], [286, 328], [329, 372],
#                  [389, 438], [439, 488], [497, 532], [533, 572], [573, 616], [617, 639], [691, 730],
#                  [731, 770], [771, 800], [801, 840], [841, 884], [908, 969], [990, 1023], [1042, 1073],
#                  [1074, 1107], [1108, 1141], [1142, 1182], [1184, 1220], [1221, 1259], [1264, 1324],
#                  [1402, 1441], [1442, 1481], [1482, 1523], [1524, 1563], [1564, 1600], [1601, 1640],
#                  [1641, 1680], [1681, 1720], [1721, 1753], [1770, 1803], [1804, 1840], [1845, 1885],
#                  [1886, 1925], [1926, 1965], [1966, 2005], [2006, 2045], [2046, 2083], [2137, 2177],
#                  [2178, 2213], [2214, 2247], [2248, 2286], [2313, 2369]]

# FTSE_fragment = [[0, 30], [30, 60], [94, 124], [124, 154], [154, 184], [189, 219], [219, 249], [286, 316], [316, 346],
#                    [389, 419], [419, 449], [449, 479], [497, 527], [533, 563], [563, 593], [691, 721], [721, 751],
#                    [751, 781], [781, 811], [811, 841], [841, 871], [908, 938], [938, 968], [990, 1020], [1042, 1072],
#                    [1074, 1104], [1108, 1138], [1184, 1214], [1214, 1244], [1264, 1294],
#                    [1402, 1432], [1432, 1462], [1462, 1492], [1492, 1522], [1524, 1554], [1564, 1594], [1594, 1624],
#                    [1624, 1654], [1654, 1684], [1684, 1714], [1714, 1744], [1770, 1800], [1800, 1830], [1845, 1875],
#                    [1875, 1905], [1905, 1935], [1935, 1965], [1965, 1995], [1995, 2025], [2025, 2055], [2137, 2167],
#                    [2167, 2197], [2214, 2244], [2244, 2274], [2313, 2343]]

# N_frag = [[14, 44], [44, 74], [74, 104], [111, 141], [141, 171], [171, 201], [201, 231], [231, 261], [278, 308],
#           [308, 338], [338, 368], [389, 419], [419, 449], [514, 544], [544, 574], [574, 604], [604, 634], [665, 695],
#           [728, 758], [769, 799], [799, 829], [829, 859], [859, 889], [889, 919], [931, 961], [961, 991], [991, 1021],
#           [1107, 1137], [1137, 1167], [1167, 1197], [1197, 1227], [1236, 1266], [1266, 1296], [1347, 1377],
#           [1503, 1533], [1533, 1563], [1563, 1593], [1593, 1623], [1623, 1653], [1653, 1683], [1683, 1713],
#           [1713, 1743], [1820, 1850], [1855, 1885], [1976, 2006], [2006, 2036], [2040, 2070], [2096, 2126],
#           [2135, 2165], [2165, 2195], [2195, 2225], [2225, 2255], [2317, 2347]]

D_frag = [[0, 30], [30, 60], [60, 90], [128, 158], [158, 188], [188, 218], [218, 248], [274, 304], [304, 334],
          [334, 364], [409, 439], [477, 507], [533, 563], [563, 593], [593, 623], [723, 753], [769, 799], [799, 829],
          [829, 859], [859, 889], [890, 920], [920, 950], [970, 1000], [1076, 1106], [1106, 1136], [1136, 1166],
          [1166, 1196], [1196, 1226], [1233, 1263], [1281, 1311], [1330, 1360], [1416, 1446], [1446, 1476],
          [1476, 1506], [1524, 1554], [1554, 1584], [1584, 1614], [1627, 1657], [1661, 1691], [1691, 1721],
          [1721, 1751], [1770, 1800], [1800, 1830], [1845, 1875], [1875, 1905], [1916, 1946], [1946, 1976],
          [1976, 2006], [2006, 2036], [2036, 2066], [2066, 2096], [2096, 2126], [2134, 2164], [2164, 2194],
          [2194, 2224], [2224, 2254], [2297, 2327], [2327, 2357]]

for f in range(len(D_frag)):
    start, end = D_frag[f][0], D_frag[f][1]
    if end - start != 30:
        print('Error!!!!!!', start, end)
    fragment = FTSE[start:end]
    fragment_dif = np.sum((fragment[:-1] - fragment[1:])**2, axis=1)
    idx = np.arange(start, end-1)
    print(idx[fragment_dif > FTSE_mean + 2.5 * FTSE_std])

print(len(D_frag))

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
# plt.show()

print(np.sum((FTSE[1341] - FTSE[1342])**2))
print(np.sum((FTSE[1342] - FTSE[1343])**2))
print(FTSE_dif[1340:1343])


