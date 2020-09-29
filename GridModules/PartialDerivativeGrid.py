import numpy as np
import math
from numpy.linalg import inv


class PartialDerivativeGrid:
    @staticmethod
    # t_sro is the total sum rules, sro is sum rules. For more info, check "PDE-Net".
    def m2q(t_sro, sro, start_index=0):
        mm = np.zeros([t_sro])
        mm[sro] = 1
        m2q_mat = np.zeros([t_sro, t_sro])
        kk = np.asarray(range(t_sro))
        kk += start_index
        for i in range(t_sro):
            m2q_mat[i, :] = kk ** i
        qq = np.matmul(np.linalg.inv(m2q_mat), mm) * math.factorial(sro)
        return qq

    # size is the matrix size. target_vector * d_mat
    @staticmethod
    def pde_1d_mat(t_sro, sro, size):
        h_ = int(np.floor(t_sro / 2))
        d_mat = np.zeros([size, size])
        for i in range(size):
            if i < h_:
                d_mat[i, :t_sro] = PartialDerivativeGrid.m2q(t_sro, sro, start_index=-i)
            elif i > size - 1 - h_:
                d_mat[i, size - t_sro:] = PartialDerivativeGrid.m2q(t_sro, sro, start_index=size - i - t_sro)
            # if i < h_:
            #     d_mat[i, :i + t_sro - h_] = PartialDerivativeGrid.m2q(i + t_sro - h_, sro, start_index=-i)
            # elif i > size - 1 - h_:
            #     d_mat[i, i - h_:] = PartialDerivativeGrid.m2q(size - i + h_, sro, start_index=-h_)
            else:
                d_mat[i, i - h_:i + t_sro - h_] = PartialDerivativeGrid.m2q(t_sro, sro, start_index=-h_)
        d_mat = np.transpose(d_mat)
        return d_mat
