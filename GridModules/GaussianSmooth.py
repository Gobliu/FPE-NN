import numpy as np
import matplotlib.pyplot as plt


class GaussianSmooth:
    @staticmethod
    def gau1d_mat(x_points, sigma):
        mat = np.zeros((x_points, x_points))
        x = np.linspace(0, 1, num=x_points, endpoint=False)
        for i in range(x_points):
            line_i = np.exp(-(x - x[i]) ** 2 / (2 * sigma ** 2))
            line_i /= sum(line_i)
            mat[i, :] = line_i
        return mat

    @staticmethod
    def gaussian1d(ip, sigma=10):
        assert ip.ndim == 1, 'Input must be a vector. Actual shape: {}'.format(ip.shape)
        x_points = len(ip)
        ip = np.expand_dims(ip, axis=1)
        smooth_mat = GaussianSmooth.gau1d_mat(x_points, sigma=sigma)
        smooth_ip = np.matmul(smooth_mat, ip)
        return smooth_ip[:, 0]

    @staticmethod
    def test():
        a = np.zeros(100)
        a[29] = 1
        a[69] = 1
        smooth_a_3 = GaussianSmooth.gaussian1d(a, sigma=3)
        smooth_a_30 = GaussianSmooth.gaussian1d(a, sigma=30)
        plt.figure()
        x = np.linspace(0, 1, num=len(a), endpoint=False)
        plt.plot(x, a, 'k-')
        plt.plot(x, smooth_a_3, 'ro')
        plt.plot(x, smooth_a_30, 'bo')
        plt.legend()
        plt.show()
