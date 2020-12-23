import numpy as np
import sys

from .PDM_NG import PDM_NG


class PxtData_List:
    def __init__(self, x=None, t=None, data=None, f_list=None):
        assert x.ndim == 1, 'Input x should be a 1D vector. Actual Shape: {}'.format(x.shape)
        assert t.ndim == 2, 'Input data should be a 2D matrix. Actual Shape: {}'.format(data.shape)
        assert data.ndim == 2, 'Input data should be a 2D matrix. Actual Shape: {}'.format(data.shape)
        assert len(x) == data.shape[1], 'The x length of x coord and data don\'t match. ' \
                                        'X length: {}, Data shape {}'.format(len(x), data.shape)

        self.pxt = np.copy(data)
        self.x = x
        self.x_points = x.shape[0]
        print('x shape', self.x_points)
        self.t = np.copy(t)
        self.id = np.arange(self.pxt.shape[0])
        self.f_list = f_list
        self.n_frag = len(f_list)

        self.train_data = None
        self.train_x = None
        self.train_y = None
        self.train_id = None
        self.train_dt = None
        self.train_t = None

        self.test_data = None
        self.test_x = None
        self.test_y = None
        self.test_id = None
        self.test_dt = None
        self.test_t = None
        print('Initializing Data.')

    def sample_train_split(self, test_range):
        # pxt_copy = np.copy(self.pxt)
        # t_copy = np.copy(self.t)

        self.train_data = []
        self.train_t = []
        self.train_id = []

        self.test_data = []
        self.test_t = []
        self.test_id = []

        for f in range(self.n_frag):
            train_start, train_end = self.f_list[f][0], self.f_list[f][1] + 1 - test_range
            test_start, test_end = self.f_list[f][1] + 1 - test_range,  self.f_list[f][1] + 1

            self.train_data.append(self.pxt[train_start:train_end])
            self.train_t.append(self.t[train_start: train_end])
            self.train_id.append(self.id[train_start: train_end])

            # print(train_start, train_end, self.t.shape)
            # print(self.pxt[train_start: train_end].shape)
            # print(self.t[train_start: train_end].shape)

            self.test_data.append(self.pxt[test_start: test_end])
            self.test_t.append(self.t[test_start: test_end])
            self.test_id.append(self.id[test_start: test_end])

            # print(self.pxt[test_start: test_end].shape)

    def get_recur_win(self, recur_win):
        # assert data.ndim == 3, 'Input data should be a 3D matrix.'
        # assert t_mat.ndim == 3, 'Input t should be a 3D matrix.'
        # d1, d2, d3 = data.shape
        # print('d1 {}, d2 {}, d3 {}'.format(d1, d2, d3))
        train_count = 0
        for item in self.train_data:
            train_count += item.shape[0]

        win_x = np.zeros((train_count, self.x_points, 1))
        win_y = np.zeros((train_count, self.x_points, recur_win))
        win_id = np.zeros((train_count,), dtype=int)
        win_t = np.zeros((train_count, 1, recur_win))
        count = 0
        for i in range(len(self.train_data)):
            d2 = self.train_data[i].shape[0]
            for j in range(d2):
                win_x[count, :, 0] = np.copy(self.train_data[i][j, ...])
                win_id[count] = np.copy(self.train_id[i][j])
                start = j - (recur_win // 2)
                for k in range(recur_win):
                    if start + k < 0:
                        win_y[count, :, k] = np.copy(self.train_data[i][abs(start+k)-1, :])
                        win_t[count, 0, k] = self.train_t[i][abs(start+k)-1, 0] - self.train_t[i][j, 0]
                    elif start + k >= d2:
                        win_y[count, :, k] = np.copy(self.train_data[i][2*d2-start-k-1, :])
                        win_t[count, :, k] = self.train_t[i][2*d2-start-k-1, :] - self.train_t[i][j, 0]
                    else:
                        win_y[count, :, k] = np.copy(self.train_data[i][start+k, :])
                        win_t[count, :, k] = self.train_t[i][start+k, 0] - self.train_t[i][j, 0]
                count += 1
        return win_x, win_t, win_y, win_id
