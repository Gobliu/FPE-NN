import numpy as np
import sys

from .PDM_NG import PDM_NG


class PxtData_NG:
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
        self.n_train_sample = 0
        self.n_test_sample = 0
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.train_x = None
        self.train_y = None
        self.train_id = None
        self.train_dt = None
        self.train_t = None
        self.test_x = None
        self.test_y = None
        self.test_id = None
        self.test_dt = None
        self.test_t = None
        print('Initializing Data.')
        print('Totally {} sample, {} time points and {} x points.'.format(self.n_sample, self.t_points, self.x_points))

    def sample_train_split_e2e(self, test_range):
        pxt_copy = np.copy(self.pxt)
        t_copy = np.copy(self.t)
        self.train_data = pxt_copy[:, :-test_range, :]
        self.train_t = t_copy[:, :-test_range, :]
        self.test_data = pxt_copy[:, -test_range:, :]
        self.test_t = t_copy[:, -test_range:, :]
        print('Totally {} time points in train, {} time points in test.'.format(self.train_data.shape[1],
                                                                                self.test_data.shape[1]))

    def sample_train_split_seq(self, test_ratio=0.2, shuffle=None):
        seq_idx = np.arange(self.n_sample)
        if shuffle:
            np.random.shuffle(seq_idx)
        train_idx = seq_idx[: - int(self.n_sample * test_ratio)]
        test_idx = seq_idx[-int(self.n_sample * test_ratio):]
        print(train_idx, test_idx)
        self.train_data = np.copy(self.pxt[train_idx])
        self.train_t = np.copy(self.t[train_idx])
        self.test_data = np.copy(self.pxt[test_idx])
        self.test_t = np.copy(self.t[test_idx])
        print('Totally {} time points in train, {} time points in test.'.format(self.train_data.shape[1],
                                                                                self.test_data.shape[1]))

    def process_for_lsq_wo_t(self, t_sro):
        print('t_points: {}'.format(self.train_data.shape[1]))
        dt = PDM_NG.pde_1d_mat(7, 1, self.train_data.shape[1]) / self.t_gap
        dt = np.transpose(dt)
        lsq_x = np.copy(self.train_data)
        lsq_y = np.zeros(self.train_data.shape)
        for sample in range(self.train_data.shape[0]):
            lsq_y[sample] = np.matmul(dt, lsq_x[sample])
        lsq_x = lsq_x.reshape(-1, self.x_points)
        lsq_y = lsq_y.reshape(-1, self.x_points)
        return lsq_x, lsq_y

    @staticmethod
    def get_recur_win(data, recur_win):
        assert data.ndim == 3, 'Input data should be a 3D matrix.'
        d1, d2, d3 = data.shape
        print('d1 {}, d2 {}, d3 {}'.format(d1, d2, d3))
        win_x = np.zeros((d1, d2 - recur_win + 1, d3, 1))
        win_y = np.zeros((d1, d2 - recur_win + 1, d3, recur_win))
        win_id = np.zeros((d1, d2 - recur_win + 1, 2), dtype=int)
        for sample in range(d1):
            for t in range(d2 - recur_win + 1):
                win_x[sample, t, :, 0] = data[sample, t + recur_win // 2, :]
                win_y[sample, t, :, :] = np.transpose(data[sample, t: t + recur_win, :])
                win_id[sample, t, :] = [sample, t + recur_win // 2]
        win_x = win_x.reshape((-1, d3, 1))
        win_y = win_y.reshape((-1, d3, recur_win))
        win_id = win_id.reshape((-1, 2))
        return win_x, win_y, win_id

    @staticmethod
    def get_recur_win_e2e(data, t_mat, recur_win):
        assert data.ndim == 3, 'Input data should be a 3D matrix.'
        assert t_mat.ndim == 3, 'Input t should be a 3D matrix.'
        d1, d2, d3 = data.shape
        print('d1 {}, d2 {}, d3 {}'.format(d1, d2, d3))
        win_x = np.zeros((d1, d2, d3, 1))
        win_y = np.zeros((d1, d2, d3, recur_win))
        win_id = np.zeros((d1, d2, 2), dtype=int)
        win_t = np.zeros((d1, d2, recur_win))
        for sample in range(d1):
            for t_idx in range(d2):
                win_x[sample, t_idx, :, 0] = np.copy(data[sample, t_idx, :])
                start = t_idx - (recur_win // 2)
                for i_ in range(recur_win):
                    # if start + i_ < 0:
                    #     win_y[sample, t_idx, :, i_] = np.copy(data[sample, 0, :])
                    #     win_t[sample, t_idx, i_] = t_mat[sample, 0, 0] - t_mat[sample, t_idx, 0]
                    # elif start + i_ >= d2:
                    #     win_y[sample, t_idx, :, i_] = np.copy(data[sample, -1, :])
                    #     win_t[sample, t_idx, i_] = t_mat[sample, -1, 0] - t_mat[sample, t_idx, 0]
                    if start + i_ < 0:
                        win_y[sample, t_idx, :, i_] = np.copy(data[sample, abs(start+i_)-1, :])
                        win_t[sample, t_idx, i_] = t_mat[sample, abs(start+i_)-1, 0] - t_mat[sample, t_idx, 0]
                    elif start + i_ >= d2:
                        win_y[sample, t_idx, :, i_] = np.copy(data[sample, 2*d2-start-i_-1, :])
                        win_t[sample, t_idx, i_] = t_mat[sample, 2*d2-start-i_-1, 0] - t_mat[sample, t_idx, 0]
                    else:
                        win_y[sample, t_idx, :, i_] = np.copy(data[sample, start + i_, :])
                        win_t[sample, t_idx, i_] = t_mat[sample, start + i_, 0] - t_mat[sample, t_idx, 0]
                win_id[sample, t_idx, :] = [sample, t_idx]
        win_x = win_x.reshape((-1, d3, 1))
        win_y = win_y.reshape((-1, d3, recur_win))
        win_id = win_id.reshape((-1, 2))
        win_t = win_t.reshape((-1, 1, recur_win))
        return win_x, win_t, win_y, win_id

    @staticmethod
    def get_recur_win_e2e_cw(data, t_mat, recur_win, cw):
        assert data.ndim == 3, 'Input data should be a 3D matrix.'
        assert t_mat.ndim == 3, 'Input t should be a 3D matrix.'
        d1, d2, d3 = data.shape
        print('d1 {}, d2 {}, d3 {}'.format(d1, d2, d3))
        win_x = np.zeros((d1, d2, d3, 1))
        win_y = np.zeros((d1, d2, d3, recur_win + cw))
        win_id = np.zeros((d1, d2, 2), dtype=int)
        win_t = np.zeros((d1, d2, recur_win + cw))
        for sample in range(d1):
            for t_idx in range(d2):
                win_x[sample, t_idx, :, 0] = np.copy(data[sample, t_idx, :])
                start = t_idx - (recur_win // 2)
                for i_ in range(recur_win):
                    # if start + i_ < 0:
                    #     win_y[sample, t_idx, :, i_] = np.copy(data[sample, 0, :])
                    #     win_t[sample, t_idx, i_] = t_mat[sample, 0, 0] - t_mat[sample, t_idx, 0]
                    # elif start + i_ >= d2:
                    #     win_y[sample, t_idx, :, i_] = np.copy(data[sample, -1, :])
                    #     win_t[sample, t_idx, i_] = t_mat[sample, -1, 0] - t_mat[sample, t_idx, 0]
                    if start + i_ < 0:
                        win_y[sample, t_idx, :, i_] = np.copy(data[sample, abs(start + i_) - 1, :])
                        win_t[sample, t_idx, i_] = t_mat[sample, abs(start + i_) - 1, 0] - t_mat[sample, t_idx, 0]
                    elif start + i_ >= d2:
                        win_y[sample, t_idx, :, i_] = np.copy(data[sample, 2 * d2 - start - i_ - 1, :])
                        win_t[sample, t_idx, i_] = t_mat[sample, 2 * d2 - start - i_ - 1, 0] - t_mat[sample, t_idx, 0]
                    else:
                        win_y[sample, t_idx, :, i_] = np.copy(data[sample, start + i_, :])
                        win_t[sample, t_idx, i_] = t_mat[sample, start + i_, 0] - t_mat[sample, t_idx, 0]
                for i_ in range(recur_win, recur_win+cw):
                    win_y[sample, t_idx, :, i_] = np.copy(data[sample, t_idx, :])
                    win_t[sample, t_idx, i_] = 0
                win_id[sample, t_idx, :] = [sample, t_idx]
        win_x = win_x.reshape((-1, d3, 1))
        win_y = win_y.reshape((-1, d3, recur_win))
        win_id = win_id.reshape((-1, 2))
        win_t = win_t.reshape((-1, 1, recur_win))
        return win_x, win_t, win_y, win_id

    def process_for_recur_net(self, recur_win=7):
        assert self.train_data is not None, 'Train data is empty.'
        assert self.valid_data is not None, 'Train data is empty.'
        assert self.test_data is not None, 'Train data is empty.'
        assert self.train_data.ndim == 3, 'Train data should be a 3D matrix. Shape: {}'.format(self.train_data.shape)
        assert self.valid_data.ndim == 3, 'Valid data should be a 3D matrix. Shape: {}'.format(self.valid_data.shape)
        assert self.test_data.ndim == 3, 'Test data should be a 3D matrix. Shape: {}'.format(self.test_data.shape)
        assert recur_win % 2 == 1, 'Expecting recur_win to be a odd integer.'
        self.train_x, self.train_y, self.train_id = self.get_recur_win(self.train_data, recur_win=recur_win)
        print('Shape of train_x: {}, train_y: {}'.format(self.train_x.shape, self.train_y.shape))
        if self.n_valid_sample != 0:
            self.valid_x, self.valid_y, self.valid_id = self.get_recur_win(self.valid_data, recur_win=recur_win)
            print('Shape of valid_x: {}, valid_y: {}'.format(self.valid_x.shape, self.valid_y.shape))
        if self.n_test_sample != 0:
            self.test_x, self.test_y, self.test_id = self.get_recur_win(self.test_data, recur_win=recur_win)
            print('Shape of test_x: {}, test_y: {}'.format(self.test_x.shape, self.test_y.shape))

    def process_for_recur_net_e2e(self, recur_win):
        assert self.train_data is not None, 'Train data is empty.'
        assert self.test_data is not None, 'Train data is empty.'
        assert self.train_data.ndim == 3, 'Train data should be a 3D matrix. Shape: {}'.format(self.train_data.shape)
        assert self.test_data.ndim == 3, 'Test data should be a 3D matrix. Shape: {}'.format(self.test_data.shape)
        assert recur_win % 2 == 1, 'Expecting recur_win to be a odd integer.'
        self.train_x, self.train_y, self.train_id, self.train_dt = self.get_recur_win_e2e(self.train_data,
                                                                                          self.train_t,
                                                                                          recur_win=recur_win)
        print('Shape of train_x: {}, train_y: {}'.format(self.train_x.shape, self.train_y.shape))
        if self.n_test_sample != 0:
            self.test_x, self.test_y, self.test_id, self.test_dt = self.get_recur_win_e2e(self.test_data,
                                                                                          self.test_t,
                                                                                          recur_win=recur_win)
            print('Shape of test_x: {}, test_y: {}'.format(self.test_x.shape, self.test_y.shape))

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
                # print(t_idx, t_idx + recur_win//2, d2-recur_win//2, new_d2)
                win_x[sample, t_idx, :, 0] = np.copy(data[sample, t_idx + recur_win//2, :])
                start = t_idx
                for i_ in range(recur_win):
                    assert 0 <= start+i_ < d2, 'Warning: index out of range: {}'.format(start+i_)
                    win_y[sample, t_idx, :, i_] = np.copy(data[sample, start + i_, :])
                    win_t[sample, t_idx, i_] = t_mat[sample, start + i_, 0] - t_mat[sample, t_idx, 0]
                win_id[sample, t_idx, :] = [sample, t_idx + recur_win//2]
        win_x = win_x.reshape((-1, d3, 1))
        win_y = win_y.reshape((-1, d3, recur_win))
        win_id = win_id.reshape((-1, 2))
        win_t = win_t.reshape((-1, 1, recur_win))
        return win_x, win_t, win_y, win_id

    @staticmethod
    def get_recur_win_back(data, t_mat, recur_win):
        assert data.ndim == 3, 'Input data should be a 3D matrix.'
        assert t_mat.ndim == 3, 'Input t should be a 3D matrix.'
        d1, d2, d3 = data.shape                 # n_sample, t_points, x_points
        print('d1 {}, d2 {}, d3 {}'.format(d1, d2, d3))
        new_d2 = d2 - recur_win + 1
        win_x = np.zeros((d1, new_d2, d3, 1))
        win_y = np.zeros((d1, new_d2, d3, recur_win))
        win_id = np.zeros((d1, new_d2, 2), dtype=int)
        win_t = np.zeros((d1, new_d2, recur_win))
        for sample in range(d1):
            for t_idx in range(recur_win-1, d2):
                win_x[sample, t_idx, :, 0] = np.copy(data[sample, t_idx, :])
                start = t_idx - recur_win + 1
                for i_ in range(recur_win):
                    assert 0 <= start+i_ < d2, 'Warning: index out of range: {}'.format(start+i_)
                    win_y[sample, t_idx, :, i_] = np.copy(data[sample, start + i_, :])
                    win_t[sample, t_idx, i_] = t_mat[sample, start + i_, 0] - t_mat[sample, t_idx, 0]
                win_id[sample, t_idx, :] = [sample, t_idx]
        win_x = win_x.reshape((-1, d3, 1))
        win_y = win_y.reshape((-1, d3, recur_win))
        win_id = win_id.reshape((-1, 2))
        win_t = win_t.reshape((-1, 1, recur_win))
        return win_x, win_t, win_y, win_id
