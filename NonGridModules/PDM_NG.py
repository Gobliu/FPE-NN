import numpy as np
import math


class PDM_NG:
    @staticmethod
    # t_sro is the total sum rules, sro is sum rules. For more info, check "PDE-Net".
    def pde_vector(dis_v, t_sro, sro):
        p_idx = np.argsort(abs(dis_v))[:t_sro]
        p_dis = dis_v[p_idx]
        # print('p_idx: ', p_idx)
        mm = np.zeros(t_sro)
        mm[sro] = 1
        m2q_mat = np.zeros([t_sro, t_sro])
        # print(p_dis)
        for i in range(t_sro):
            m2q_mat[i, :] = p_dis ** i / math.factorial(i)
        # print(m2q_mat)
        qq = np.matmul(np.linalg.inv(m2q_mat), mm)
        co_v = np.zeros(dis_v.shape)
        co_v[p_idx] = qq
        return co_v

    # size is the matrix size. target_vector * d_mat
    @staticmethod
    def pde_1d_mat(x_coord, t_sro, sro):
        """ Generate a derivative matrix M. If a given vector A (shape [-1, 1]) multiple with M, will get the sro_th
         derivative of A (shape [-1, 1] """
        size = x_coord.shape[0]
        dis_matrix = x_coord.reshape(-1, 1)
        dis_matrix = np.repeat(dis_matrix, size, axis=1)
        dis_matrix = dis_matrix - x_coord
        pde_mat = np.zeros(dis_matrix.shape)
        # print(dis_matrix)
        for idx in range(size):
            # print(idx, dis_matrix[:, idx])
            pde_mat[:, idx] = PDM_NG.pde_vector(dis_matrix[:, idx], t_sro, sro)

        return pde_mat


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True)
    x = np.random.uniform(size=1000) * math.pi * 2
    # ======== sorting is not required
    # x.sort()
    # ========
    # x = x.reshape((2, -1))
    dx_mat = PDM_NG.pde_1d_mat(x, 20, 1)
    dxx_mat = PDM_NG.pde_1d_mat(x, 20, 2)
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
