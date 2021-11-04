import sys
import numpy as np

from .PartialDerivativeGrid import PartialDerivativeGrid as PDeGrid


class FokkerPlankForward:
    """
    Forward solver of Fokker-Plank Equation
    """
    @staticmethod
    def forward(g, h, p0, x_gap, t_gap, t_points, direction=1, rk=1, t_sro=7):
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

            else:
                sys.exit('The order of Runge-Kutta method is out of scope.')
            p_dt[t-1] = k1
        if direction == -1:
            pt = np.flip(pt, axis=0)
        return pt
