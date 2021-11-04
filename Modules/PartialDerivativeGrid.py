import numpy as np
import math
from numpy.linalg import inv


class PartialDerivativeGrid:
    """
    Generate the derivative matrix with respective to variable x or t.
    The variable points (x/t) has to be even distributed (uniform gap).
    """
    @staticmethod
    def m2q(t_sro, sro, start_index=0):
        """
        Local function used in pde_1d_mat below
        """
        mm = np.zeros([t_sro])
        mm[sro] = 1
        m2q_mat = np.zeros([t_sro, t_sro])
        kk = np.asarray(range(t_sro))
        kk += start_index
        for i in range(t_sro):
            m2q_mat[i, :] = kk ** i
        qq = np.matmul(np.linalg.inv(m2q_mat), mm) * math.factorial(sro)
        return qq

    @staticmethod
    def pde_1d_mat(t_sro, sro, size):
        """"
        :param t_sro: total sum rules, check "PDE-Net" paper for more detail
        :param sro: total sum , check "PDE-Net" paper for more detail
        :param size: the vector length of the variable points, can be x or t
        :return: derivative matrix pde_mat. If a given vector A (shape [-1, 1]) multiple with pde_mat, will get the
         sro_th derivative of A (shape [-1, 1])
        """
        h_ = int(np.floor(t_sro / 2))
        d_mat = np.zeros([size, size])
        for i in range(size):
            if i < h_:
                d_mat[i, :t_sro] = PartialDerivativeGrid.m2q(t_sro, sro, start_index=-i)
            elif i > size - 1 - h_:
                d_mat[i, size - t_sro:] = PartialDerivativeGrid.m2q(t_sro, sro, start_index=size - i - t_sro)
            else:
                d_mat[i, i - h_:i + t_sro - h_] = PartialDerivativeGrid.m2q(t_sro, sro, start_index=-h_)
        d_mat = np.transpose(d_mat)
        return d_mat
