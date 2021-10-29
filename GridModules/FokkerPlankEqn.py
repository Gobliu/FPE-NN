import sys
from scipy import signal
import numpy as np

from .PartialDerivativeGrid import PartialDerivativeGrid as PDeGrid


class FokkerPlankForward:
    @staticmethod
    def ghx2pxt(g, h, p0, x_gap, t_gap, t_points, direction=1, rk=1, t_sro=7):
        assert g.ndim == 1, "gx should be a vector. Shape of gx: {}".format(g.shape)
        assert h.ndim == 1, "hx should be a vector. Shape of hx: {}".format(h.shape)
        assert p0.ndim == 1, "P0 should be a vector. Shape of P: {}".format(p0.shape)
        x_points = len(p0)
        dx = PDeGrid.pde_1d_mat(t_sro, 1, x_points) / x_gap
        dxx = PDeGrid.pde_1d_mat(t_sro, 2, x_points) / x_gap**2
        assert isinstance(rk, int), "The order of Runge-Kutta method must be integer."
        assert rk <= 4, "The order of Runge-Kutta method is out of scope."
        pt = np.zeros((t_points, x_points))
        p_dt = np.zeros((t_points-1, x_points))
        pt[0] = p0
        for t in range(1, t_points):
            # pt[pt < 0] = 0
            pre_p = pt[t-1]
            if rk == 1:
                k1 = np.matmul(g * pre_p, dx) + np.matmul(h * pre_p, dxx)
                pt[t] = pre_p + k1 * t_gap * direction
            elif rk == 4:
                k1 = np.matmul(g * pre_p, dx) + np.matmul(h * pre_p, dxx)
                y2 = pre_p + k1 * t_gap / 2 * direction
                k2 = np.matmul(g * y2, dx) + np.matmul(h * y2, dxx)
                y3 = pre_p + k2 * t_gap / 2 * direction
                k3 = np.matmul(g * y3, dx) + np.matmul(h * y3, dxx)
                y4 = pre_p + k3 * t_gap * direction
                k4 = np.matmul(g * y4, dx) + np.matmul(h * y4, dxx)
                pt[t] = pre_p + (k1 + 2 * k2 + 2 * k3 + k4) * t_gap / 6 * direction

                # print('k1', k1)
                # print('k2', k2)
                # print('k3', k3)
                # print('k4', k4)
            else:
                sys.exit('The order of Runge-Kutta method is out of scope.')
            p_dt[t-1] = k1
        if direction == -1:
            pt = np.flip(pt, axis=0)
        return pt

    @staticmethod
    def ghx2pxt_pad_smooth(g, h, p0, x_gap, t_gap, t_points, pad_size=5, direction=1, t_sro=7):
        assert g.ndim == 1, "gx should be a vector. Shape of gx: {}".format(g.shape)
        assert h.ndim == 1, "hx should be a vector. Shape of hx: {}".format(h.shape)
        assert p0.ndim == 1, "P0 should be a vector. Shape of P: {}".format(p0.shape)
        x_points = len(p0) + 2*pad_size
        dx = PDeGrid.pde_1d_mat(t_sro, 1, x_points) / x_gap
        dxx = PDeGrid.pde_1d_mat(t_sro, 2, x_points) / x_gap**2

        exp_g = np.zeros(x_points)
        exp_g[pad_size: -pad_size] = g
        exp_h = np.zeros(x_points)
        exp_h[pad_size: -pad_size] = h
        pt = np.zeros((t_points, x_points))
        p_dt = np.zeros((t_points-1, x_points))
        pt[0, pad_size: -pad_size] = p0
        for t in range(1, t_points):
            pt[t-1, :5] = 0
            pt[t-1, -5:] = 0
            # pt[pt < 0] = 0
            pt[t-1] = signal.savgol_filter(pt[t-1], 7, 2)
            pre_p = pt[t-1]

            k1 = np.matmul(exp_g * pre_p, dx) + np.matmul(exp_h * pre_p, dxx)
            pt[t] = pre_p + k1 * t_gap * direction
            p_dt[t-1] = k1
        if direction == -1:
            pt = np.flip(pt, axis=0)
        final_pt = pt[:, pad_size: -pad_size]
        return final_pt

    @staticmethod
    def ghx2pxt_small_step(g, h, p0, x_gap, t_gap, t_points, step=1, t_sro=7):
        """too few steps, like 2 or 3, might make it worse."""
        assert g.ndim == 1, "gx should be a vector. Shape of gx: {}".format(g.shape)
        assert h.ndim == 1, "hx should be a vector. Shape of hx: {}".format(h.shape)
        assert p0.ndim == 1, "P0 should be a vector. Shape of P: {}".format(p0.shape)
        x_points = len(p0)
        dx = PDeGrid.pde_1d_mat(t_sro, 1, x_points) / x_gap
        dxx = PDeGrid.pde_1d_mat(t_sro, 2, x_points) / x_gap**2
        pt = np.zeros((t_points, x_points))
        t_step_gap = t_gap / step
        pt[0] = p0
        pre_p = np.copy(p0)     # must copy, otherwise change p0
        for t in range(1, t_points * step):
            k1 = np.matmul(g * pre_p, dx) + np.matmul(h * pre_p, dxx)
            pre_p += k1 * t_step_gap
            pre_p[pre_p < 0] = 0
            if t % step == 0:
                pt[t // step] = pre_p
        return pt

    @staticmethod
    def ghx2pxt_euler(g, h, p0, x_gap, t_gap, t_points, pad_size=5, direction=1, t_sro=7):
        assert g.ndim == 1, "gx should be a vector. Shape of gx: {}".format(g.shape)
        assert h.ndim == 1, "hx should be a vector. Shape of hx: {}".format(h.shape)
        assert p0.ndim == 1, "P0 should be a vector. Shape of P: {}".format(p0.shape)
        x_points = len(p0) + 2*pad_size
        dx = PDeGrid.pde_1d_mat(t_sro, 1, x_points) / x_gap
        dxx = PDeGrid.pde_1d_mat(t_sro, 2, x_points) / x_gap**2

        exp_g = np.zeros(x_points)
        exp_g[pad_size: -pad_size] = g
        exp_h = np.zeros(x_points)
        exp_h[pad_size: -pad_size] = h
        pt = np.zeros((t_points, x_points))
        p_dt = np.zeros((t_points-1, x_points))
        pt[0, pad_size: -pad_size] = p0
        pt[0, :5] = 0
        pt[0, -5:] = 0
        pt[0] = signal.savgol_filter(pt[0], 7, 2)
        k1 = np.matmul(exp_g * pt[0], dx) + np.matmul(exp_h * pt[0], dxx)
        for t in range(1, t_points):
            pt[t] = pt[0] + k1 * t * t_gap * direction
        if direction == -1:
            pt = np.flip(pt, axis=0)
        final_pt = pt[:, pad_size: -pad_size]
        return final_pt

    @staticmethod
    def ghxt2pxt(g_xt, h_xt, p0, x_gap, t_gap, rk=1, t_sro=7):
        assert p0.ndim == 1, "P0 should be a vector. Shape of P: {}".format(p0.shape)
        assert g_xt.ndim == 2, "g_x_t should be a 2D matrix. Shape of g: {}".format(g_xt.shape)
        assert h_xt.ndim == 2, "h_x_t should be a 2D matrix. Shape of h: {}".format(h_xt.shape)
        assert g_xt.shape[1] == len(p0), "The shapes of g and p0 are different. p0: {}, g: {}".format(p0.shape,
                                                                                                      g_xt.shape)
        assert g_xt.shape == h_xt.shape, "The shapes of g and h are different. g: {}, h: {}".format(g_xt.shape,
                                                                                                    h_xt.shape)
        assert isinstance(rk, int), "The order of Runge-Kutta method must be integer."
        assert rk <= 4, "The order of Runge-Kutta method is out of scope."

        t_points, x_points = g_xt.shape
        dx = PDeGrid.pde_1d_mat(t_sro, 1, x_points) / x_gap
        # dxx = PDe.pde_mat(t_sro, 2, x_points) / x_gap**2
        dxx = np.matmul(dx, dx)
        pt = np.zeros((t_points, x_points))
        pt[0] = p0

        for t in range(1, t_points):
            pre_p = pt[t-1]
            g = g_xt[t]
            h = h_xt[t]
            if rk == 1:
                k1 = np.matmul(g * pre_p, dx) + np.matmul(h * pre_p, dxx)
                pt[t] = pre_p + k1 * t_gap
            elif rk == 2:
                k1 = np.matmul(g * pre_p, dx) + np.matmul(h * pre_p, dxx)
                y2 = pre_p + k1 * t_gap / 2
                k2 = np.matmul(g * y2, dx) + np.matmul(h * y2, dxx)
                pt[t] = pre_p + k2 * t_gap
            elif rk == 3:
                k1 = np.matmul(g * pre_p, dx) + np.matmul(h * pre_p, dxx)
                y2 = pre_p + k1 * t_gap / 3
                k2 = np.matmul(g * y2, dx) + np.matmul(h * y2, dxx)
                y3 = pre_p + k2 * t_gap * 2 / 3
                k3 = np.matmul(g * y3, dx) + np.matmul(h * y3, dxx)
                pt[t] = pre_p + (k1 + 3 * k3) * t_gap / 4
            elif rk == 4:
                k1 = np.matmul(g * pre_p, dx) + np.matmul(h * pre_p, dxx)
                y2 = pre_p + k1 * t_gap / 2
                k2 = np.matmul(g * y2, dx) + np.matmul(h * y2, dxx)
                y3 = pre_p + k2 * t_gap / 2
                k3 = np.matmul(g * y3, dx) + np.matmul(h * y3, dxx)
                y4 = pre_p + k3 * t_gap
                k4 = np.matmul(g * y4, dx) + np.matmul(h * y4, dxx)
                pt[t] = pre_p + (k1 + 2 * k2 + 2 * k3 + k4) * t_gap / 6
            else:
                sys.exit('The order of Runge-Kutta method is out of scope.')
            # if t % 10 == 0:
            #     pt[t] /= np.sum(pt[t])
            #     pt[t] *= x_points
            # pt[pt < 0] = 0
        return pt
