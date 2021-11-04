import numpy as np
import math


class PartialDerivativeNonGrid:
    """
    To generate the derivative matrix with respective to variable x or t.
    The variable points (x/t) is not necessarily to be even distributed (uniform gap) or sorted.
    Demonstrated by the main function below.
    """
    @staticmethod
    def pde_vector(dis_v, t_sro, sro):
        """
        Local function used by pde_1d_mat below
        """
        p_idx = np.argsort(abs(dis_v))[:t_sro]
        p_dis = dis_v[p_idx]

        mm = np.zeros(t_sro)
        mm[sro] = 1
        m2q_mat = np.zeros([t_sro, t_sro])

        for i in range(t_sro):
            m2q_mat[i, :] = p_dis ** i / math.factorial(i)

        qq = np.matmul(np.linalg.inv(m2q_mat), mm)
        co_v = np.zeros(dis_v.shape)
        co_v[p_idx] = qq
        return co_v

    @staticmethod
    def pde_1d_mat(x_coord, t_sro, sro):
        """
        :param x_coord: the vector of the variable points, can be x or t
        :param t_sro: total sum rules, check "PDE-Net" paper for more detail
        :param sro: total sum , check "PDE-Net" paper for more detail
        :return: derivative matrix pde_mat. If a given vector A (shape [-1, 1]) multiple with pde_mat, will get the
         sro_th derivative of A (shape [-1, 1])
        """
        size = x_coord.shape[0]
        dis_matrix = x_coord.reshape(-1, 1)
        dis_matrix = np.repeat(dis_matrix, size, axis=1)
        dis_matrix = dis_matrix - x_coord
        pde_mat = np.zeros(dis_matrix.shape)
        for idx in range(size):
            pde_mat[:, idx] = PartialDerivativeNonGrid.pde_vector(dis_matrix[:, idx], t_sro, sro)

        return pde_mat


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True)
    x = np.random.uniform(size=1000) * math.pi * 2
    dx_mat = PartialDerivativeNonGrid.pde_1d_mat(x, 20, 1)
    dxx_mat = PartialDerivativeNonGrid.pde_1d_mat(x, 20, 2)
    y = np.zeros((2, x.shape[0]))
    y[0, :] = np.sin(x)
    y[1, :] = np.cos(x)
    dy_dx = np.matmul(y, dx_mat)
    dy_dxx = np.matmul(y, dxx_mat)
    plt.figure()
    plt.plot(x, y[0, :], 'ko', label='y')
    plt.plot(x, dy_dx[0, :], 'b.', label='dy/dx')
    plt.plot(x, dy_dxx[0, :], 'r+', label='dy/dxx')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x, y[1, :], 'ko', label='y')
    plt.plot(x, dy_dx[1, :], 'b.', label='dy/dx')
    plt.plot(x, dy_dxx[1, :], 'r+', label='dy/dxx')
    plt.legend()
    plt.show()
