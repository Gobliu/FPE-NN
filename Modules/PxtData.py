import numpy as np


class PxtData:
    """
    Process the distribution sequences to generate input and target-output pairs for network training
    """
    def __init__(self, x=None, t=None, data=None):
        assert x.ndim == 1, 'Input x should be a 1D vector. Actual Shape: {}'.format(x.shape)
        assert t.ndim == 3, 'Input data should be a 3D matrix. Actual Shape: {}'.format(data.shape)
        assert data.ndim == 3, 'Input data should be a 3D matrix. Actual Shape: {}'.format(data.shape)
        assert len(x) == data.shape[2], 'The x length of x coord and data don\'t match. ' \
                                        'X length: {}, Data shape {}'.format(len(x), data.shape)

        self.pxt = np.copy(data)
        self.x = x
        self.t = t
        self.n_sample, self.t_points, self.x_points = data.shape

    @staticmethod
    def get_recur_win_center(data, t_mat, recur_win):
        assert data.ndim == 3, 'Input data should be a 3D matrix.'
        assert t_mat.ndim == 3, 'Input t should be a 3D matrix.'
        d1, d2, d3 = data.shape                 # n_sample, t_points, x_points
        print('d1 {}, d2 {}, d3 {}'.format(d1, d2, d3))
        new_d2 = d2 - 2*(recur_win // 2)
        win_x = np.zeros((d1, new_d2, d3, 1))
        win_y = np.zeros((d1, new_d2, d3, recur_win))
        win_id = np.zeros((d1, new_d2, 2), dtype=int)
        win_t = np.zeros((d1, new_d2, recur_win))
        for sample in range(d1):
            for t_idx in range(new_d2):
                win_x[sample, t_idx, :, 0] = np.copy(data[sample, t_idx + recur_win//2, :])
                start = t_idx
                for i_ in range(recur_win):
                    assert 0 <= start+i_ < d2, 'Warning: index out of range: {}'.format(start+i_)
                    win_y[sample, t_idx, :, i_] = np.copy(data[sample, start + i_, :])
                    win_t[sample, t_idx, i_] = t_mat[sample, start + i_, 0] - t_mat[sample, t_idx + recur_win//2, 0]
                win_id[sample, t_idx, :] = [sample, t_idx + recur_win//2]
        win_x = win_x.reshape((-1, d3, 1))
        win_y = win_y.reshape((-1, d3, recur_win))
        win_id = win_id.reshape((-1, 2))
        win_t = win_t.reshape((-1, 1, recur_win))
        return win_x, win_t, win_y, win_id
