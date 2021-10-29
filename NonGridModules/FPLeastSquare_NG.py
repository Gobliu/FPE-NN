import numpy as np
from numpy.linalg import inv
from .PDM_NG import PDM_NG
import sys


class FPLeastSquareNG:
    def __init__(self, x_coord, t_sro):
        self.x_coord = x_coord
        self.x_points = x_coord.shape[0]
        self.t_sro = t_sro
        self.dx = PDM_NG.pde_1d_mat(x_coord, t_sro, sro=1)
        self.dxx = PDM_NG.pde_1d_mat(x_coord, t_sro, sro=2)

    def lsq_wo_t_list(self, pxt_list, t_list):
        count = 0
        for item in pxt_list:
            count += item.shape[0]
        print('total distributions {}'.format(count))
        p_mat = np.zeros((count, self.x_points, 4))

        count = 0
        for s in range(len(pxt_list)):
            t_coord = t_list[s][:, 0]
            # print(t_coord.shape)
            dt = PDM_NG.pde_1d_mat(t_coord, self.t_sro, sro=1)
            # print(dt)
            count_end = count + pxt_list[s].shape[0]
            p_mat[count: count_end, :, 0] = np.matmul(pxt_list[s].transpose(), dt).transpose()
            p_mat[count: count_end, :, 1] = pxt_list[s]
            p_mat[count: count_end, :, 2] = np.matmul(pxt_list[s], self.dx)
            p_mat[count: count_end, :, 3] = np.matmul(pxt_list[s], self.dxx)

        # ======== check whether some points of x always have very small p
        sum_p = np.sum(p_mat[:, :, 1], axis=0)
        print(sum_p.shape)
        print('Sum_p min: {} pos: {}, max {}, pos {}'.format(min(sum_p), self.x_coord[np.argmin(sum_p)],
                                                             max(sum_p), self.x_coord[np.argmax(sum_p)]))

        # for pos in range(self.x_points):
        #     assert sum_p[pos] != 0, 'P(x) is always zeros at pos {}, {}.'.format(pos, sum_p[pos])

        abh_mat = np.zeros([3, self.x_points])
        sum_err = 0
        for x in range(self.x_points):
            # print(x, sum_p[x])
            if sum_p[x] == 0:
                abh_mat[:, x] = 0
                continue
            mat_b = p_mat[:, x, :1]     # mat_b shape:(time, 1)
            mat_a = p_mat[:, x, 1:]
            # print(x)
            abh = np.matmul(np.matmul(inv(np.matmul(mat_a.transpose(), mat_a)), mat_a.transpose()), mat_b)
            abh_mat[:, x] = abh[:, 0]   # abh shape: (3,1)
            cal_b = np.matmul(mat_a, abh)
            sum_err += np.sum((cal_b - mat_b)**2)
        print('Total error: {}'.format(sum_err))
        hh = abh_mat[2, :]
        dh_dx = np.matmul(hh, self.dx)
        gg = abh_mat[1, :] - dh_dx

        return gg, hh, dt, p_mat

    def lsq_wo_t(self, pxt, t):
        print('Initialising parameters with Linear Least Square...')
        sample_no, t_points, _ = pxt.shape
        assert _ == self.x_points, "Shape not consistent. x_points {}, pxt shape {}".format(self.x_points, pxt.shape)
        p_mat = np.zeros((sample_no, t_points, self.x_points, 4))
        for sample_idx in range(sample_no):
            t_coord = t[sample_idx, :, 0]       # must be 1-d
            # print(self.x_coord.shape, t_coord.shape)
            dt = PDM_NG.pde_1d_mat(t_coord, self.t_sro, sro=1)
            # print(pxt[sample_idx, ...].shape, self.dx.shape)
            p_mat[sample_idx, :, :, 0] = np.matmul(pxt[sample_idx, ...].transpose(), dt).transpose()
            p_mat[sample_idx, :, :, 1] = pxt[sample_idx, ...]
            p_mat[sample_idx, :, :, 2] = np.matmul(pxt[sample_idx, ...], self.dx)
            p_mat[sample_idx, :, :, 3] = np.matmul(pxt[sample_idx, ...], self.dxx)
        p_mat = p_mat.reshape((-1, self.x_points, 4))

        # ======== check whether some points of x always have very small p
        sum_p = np.sum(p_mat[:, :, 1], axis=0)
        print(sum_p.shape)
        print('Sum_p min: {} pos: {}, max {}, pos {}'.format(min(sum_p), self.x_coord[np.argmin(sum_p)],
                                                             max(sum_p), self.x_coord[np.argmax(sum_p)]))

        for pos in range(self.x_points):
            assert sum_p[pos] != 0, 'P(x) is always zeros at pos {}, {}.'.format(pos, sum_p[pos])

        abh_mat = np.zeros([3, self.x_points])
        sum_err = 0
        for x in range(self.x_points):
            mat_b = p_mat[:, x, :1]     # mat_b shape:(time, 1)
            # mat_b = np.expand_dims(mat_b, axis=1)   # mat_b shape:(time, 1)
            mat_a = p_mat[:, x, 1:]
            abh = np.matmul(np.matmul(inv(np.matmul(mat_a.transpose(), mat_a)), mat_a.transpose()), mat_b)
            abh_mat[:, x] = abh[:, 0]   # abh shape: (3,1)
            cal_b = np.matmul(mat_a, abh)
            sum_err += np.sum((cal_b - mat_b)**2)
        print('Total error: {}'.format(sum_err))
        hh = abh_mat[2, :]
        dh_dx = np.matmul(hh, self.dx)
        gg = abh_mat[1, :] - dh_dx

        return gg, hh, dt, p_mat


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     data = np.load('../Pxt/OU_0.npz')
#     x = data['x']
#     t = data['t']
#     true_pxt = data['true_pxt']
#     noisy_pxt = data['noisy_pxt']
#     print(x.shape)
#     # print(t)
#     # print(true_pxt)
#     LLS = FPLeastSquare_NG(x, t_sro=7)
#     cal_g, cal_h = LLS.lsq_wo_t(noisy_pxt, t)

    # plt.figure(figsize=[9, 6])
    # plt.plot(x, 2.86*x, 'k-', label='g_true')
    # plt.plot(x, cal_g, 'r*', label='g_cal')
    # plt.legend()
    # plt.show()
    #
    # plt.figure(figsize=[9, 6])
    # plt.plot(x, 0.0013 * np.ones(x.shape), 'k-', label='h_true')
    # plt.plot(x, cal_h, 'r*', label='h_cal')
    # plt.legend()
    # plt.show()

