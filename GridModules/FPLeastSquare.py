import numpy as np
from numpy.linalg import inv
import sys


class FPLeastSquare:
    def __init__(self, x_points, dx, dxx):
        self.x_points = x_points
        self.dx = dx
        self.dxx = dxx

    def lsq_wo_t(self, pp, pp_dt):
        print('Initialising parameters with Linear Least Square...')
        p_mat = np.zeros([pp.shape[0], self.x_points, 4])      # time, position and 4 values

        p_mat[:, :, 0] = pp_dt
        p_mat[:, :, 1] = pp
        p_mat[:, :, 2] = np.matmul(pp, self.dx)
        p_mat[:, :, 3] = np.matmul(pp, self.dxx)

        sum_p = np.sum(p_mat[:, :, 1], axis=0)
        # print(sum_p)
        print('Sum_p max: {} pos: {}, min {}, pos {}'.format(max(sum_p), np.argmax(sum_p)/self.x_points,
                                                             min(sum_p), np.argmin(sum_p)/self.x_points))

        for pos in range(self.x_points):
            assert sum_p[pos] != 0, 'P(x) is always zeros at pos {}, {}.'.format(pos, sum_p[pos])

        abh_mat = np.zeros([3, self.x_points])
        sum_err = 0
        for x in range(self.x_points):
            mat_b = p_mat[:, x, 0]
            mat_b = np.expand_dims(mat_b, axis=1)   # mat_b shape:(time, 1)
            mat_a = p_mat[:, x, 1:]
            abh = np.matmul(np.matmul(inv(np.matmul(mat_a.transpose(), mat_a)), mat_a.transpose()), mat_b)
            abh_mat[:, x] = abh[:, 0]   # abh shape: (3,1)
            cal_b = np.matmul(mat_a, abh)
            sum_err += np.sum((cal_b - mat_b)**2)
        print('Total error: {}'.format(sum_err))
        hh = abh_mat[2, :]
        dev_h = np.matmul(hh, self.dx)
        gg = abh_mat[1, :] - dev_h

        # print(p_mat[0, :10, :])
        # print(p_mat[1, :10, :])
        # print(p_mat[3, :10, :])
        # print(abh_mat[:, :10].transpose())

        return gg, hh, p_mat, abh_mat, dev_h

    def lsq_w_t(self, pp, pp_dt, time):
        print('Initialising parameters with Linear Least Square...')
        p_mat = np.zeros([pp.shape[0], self.x_points, 7])

        p_mat[:, :, 0] = pp_dt
        p_mat[:, :, 1] = pp
        p_mat[:, :, 2] = pp * time
        p_mat[:, :, 3] = np.matmul(pp, self.dx)
        p_mat[:, :, 4] = np.matmul(pp, self.dx) * time
        p_mat[:, :, 5] = np.matmul(pp, self.dxx)
        p_mat[:, :, 6] = np.matmul(pp, self.dxx) * time

        sum_p = np.sum(p_mat[:, :, 1], axis=0)
        print('Sum_p max: {} pos: {}, min {}, pos {}'.format(max(sum_p), np.argmax(sum_p) / self.x_points,
                                                             min(sum_p), np.argmin(sum_p) / self.x_points))
        a2h_mat = np.zeros([6, self.x_points])
        sum_err = 0
        for x in range(self.x_points):
            mat_b = p_mat[:, x, 0]
            mat_b = np.expand_dims(mat_b, axis=1)
            mat_a = p_mat[:, x, 1:]
            # print(np.shape(mat_a), np.shape(mat_b))
            a2h = np.matmul(np.matmul(inv(np.matmul(mat_a.transpose(), mat_a)), mat_a.transpose()), mat_b)
            # print(np.shape(abh), np.shape(abh[:,0]))
            a2h_mat[:, x] = a2h[:, 0]
            cal_b = np.matmul(mat_a, a2h)
            sum_err += np.sum((cal_b - mat_b) ** 2)
        print('Total error: {}'.format(sum_err))
        print('Shape of abh_mat: {}'.format(np.shape(a2h_mat)))
        h0 = a2h_mat[4, :]
        h1 = a2h_mat[5, :]
        # print(hh)
        dev_h0 = np.matmul(h0, self.dx)
        dev_h1 = np.matmul(h1, self.dxx)
        # print(np.shape(dev_h))
        g0 = a2h_mat[2, :] - dev_h0
        g1 = a2h_mat[3, :] - dev_h1
        print('g0 mean: {}, g1 mean: {}, h0 mean: {}, h1 mean: {}'.format(np.mean(g0), np.mean(g1),
                                                                          np.mean(h0), np.mean(h1)))
        return g0, g1, h0, h1

    def test_wt(self, g0, g1, h0, h1, x, y, t, k, t_gap):
        g0 = np.expand_dims(g0, axis=0)
        g1 = np.expand_dims(g1, axis=0)
        h0 = np.expand_dims(h0, axis=0)
        h1 = np.expand_dims(h1, axis=0)
        # print('Shapes:', g0.shape, x.shape)
        g = g0 + g1 * t
        h = h0 + h1 * t
        gx = g * x
        hx = h * x
        pred_k = np.matmul(gx, self.dx) + np.matmul(hx, self.dxx)
        # print(pred_k[0, :])
        error_k = np.sum((pred_k - k)[0, :] ** 2)
        pred_y = x + pred_k * t_gap
        error_y = np.sum((pred_y - y)**2)
        # print('Error of next y:{} and next k: {}.'.format(error_y, error_k))
        # return error
