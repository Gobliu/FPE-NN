import numpy as np
from numpy.linalg import inv
from .PartialDerivativeNonGrid import PartialDerivativeNonGrid as pd_ng


class FPLeastSquare:
    """
    Calculate gh from given pxt with linear least square method
    """
    def __init__(self, x_coord, t_sro):
        """
        self.dx and self.dxx are derivative matrices for x vector.
        x vector no need to be even distributed or sorted.
        """
        self.x_coord = x_coord
        self.x_points = x_coord.shape[0]
        self.t_sro = t_sro
        self.dx = pd_ng.pde_1d_mat(x_coord, t_sro, sro=1)
        self.dxx = pd_ng.pde_1d_mat(x_coord, t_sro, sro=2)

    def lsq_wo_t(self, pxt, t):
        """
        Linear least square function where pxt and t are provided separately.
        t no need to be even distributed or sorted.
        """
        print('Initialising parameters with Linear Least Square...')
        sample_no, t_points, _ = pxt.shape
        assert _ == self.x_points, "Shape not consistent. x_points {}, pxt shape {}".format(self.x_points, pxt.shape)
        p_mat = np.zeros((sample_no, t_points, self.x_points, 4))
        for sample_idx in range(sample_no):
            t_coord = t[sample_idx, :, 0]       # must be 1-d
            dt = pd_ng.pde_1d_mat(t_coord, self.t_sro, sro=1)
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
