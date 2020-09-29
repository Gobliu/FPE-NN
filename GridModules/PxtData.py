import numpy as np
import sys

sys.path.insert(1, '/home/liuwei/pyModules')
from PartialDerivativeGrid import PartialDerivativeGrid as PDeGrid


class PxtData:
    def __init__(self, t_gap=None, x=None, data=None, time=None):
        assert t_gap is not None, 'Please input t_gap.'
        assert x.ndim == 1, 'Input x should be a 1D vector. Actual Shape: {}'.format(x.shape)
        assert data.ndim == 3, 'Input data should be a 3D matrix. Actual Shape: {}'.format(data.shape)
        assert len(x) == data.shape[2], 'The x length of x coord and data don\'t match. ' \
                                        'X length: {}, Data shape {}'.format(len(x), data.shape)
        self.pxt = data
        self.x = x
        self.t_gap = t_gap
        self.time = time
        self.n_sample, self.t_points, self.x_points = data.shape
        self.n_train_sample = 0
        self.n_valid_sample = 0
        self.n_test_sample = 0
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.train_x = None
        self.train_y = None
        self.train_id = None
        self.train_dt = None
        self.train_t = None
        self.valid_x = None
        self.valid_y = None
        self.valid_id = None
        self.valid_dt = None
        self.valid_t = None
        self.test_x = None
        self.test_y = None
        self.test_id = None
        self.test_dt = None
        self.test_t = None
        print('Initializing Data.')
        print('Totally {} sample, {} time points and {} x points.'.format(self.n_sample, self.t_points, self.x_points))

    def whole_train_split(self, valid_ratio=0.2, test_ratio=0.2, seed=2012, shuffle=False):
        pxt_copy = np.copy(self.pxt)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(pxt_copy)

        valid_cutoff = int(self.n_sample * valid_ratio)
        test_cutoff = int(self.n_sample * test_ratio)
        self.test_data = pxt_copy[:test_cutoff]
        self.valid_data = pxt_copy[test_cutoff: test_cutoff + valid_cutoff]
        self.train_data = pxt_copy[valid_cutoff + test_cutoff:]

        self.n_train_sample = self.train_data.shape[0]
        self.n_valid_sample = self.valid_data.shape[0]
        self.n_test_sample = self.test_data.shape[0]
        print('Totally {} sample in train, {} sample in valid, {} sample in test.'.format(self.n_train_sample,
                                                                                          self.n_valid_sample,
                                                                                          self.n_test_sample))

    def sample_train_split(self, valid_ratio, test_ratio, recur_win):
        half_win = recur_win // 2
        pxt_copy = np.copy(self.pxt)
        valid_cutoff = int(self.t_points * valid_ratio)
        test_cutoff = int(self.t_points * test_ratio)
        train_start = max(0, valid_cutoff - half_win)
        train_end = min(self.t_points, self.t_points - test_cutoff + half_win)
        self.valid_data = pxt_copy[:, 0: valid_cutoff, :]
        self.train_data = pxt_copy[:, train_start: train_end, :]
        self.test_data = pxt_copy[:, self.t_points - test_cutoff:, :]
        print('Totally {} time points in valid, {} time points in test.'.format(valid_cutoff, test_cutoff))

    def sample_train_split_e2e(self, valid_ratio, test_ratio, recur_win):
        pxt_copy = np.copy(self.pxt)
        valid_cutoff = int(self.t_points * valid_ratio)
        test_cutoff = int(self.t_points * test_ratio)
        self.valid_data = pxt_copy[:, 0: valid_cutoff, :]
        self.train_data = pxt_copy[:, valid_cutoff:self.t_points - test_cutoff, :]
        self.test_data = pxt_copy[:, self.t_points - test_cutoff:, :]
        print('Totally {} time points in valid, {} time points in test.'.format(valid_cutoff, test_cutoff))

    def process_for_lsq_wo_t(self):
        print('t_points: {}'.format(self.train_data.shape[1]))
        dt = PDeGrid.pde_1d_mat(7, 1, self.train_data.shape[1]) / self.t_gap
        dt = np.transpose(dt)
        lsq_x = np.copy(self.train_data)
        lsq_y = np.zeros(self.train_data.shape)
        for sample in range(self.train_data.shape[0]):
            lsq_y[sample] = np.matmul(dt, lsq_x[sample])
        lsq_x = lsq_x.reshape(-1, self.x_points)
        lsq_y = lsq_y.reshape(-1, self.x_points)
        return lsq_x, lsq_y

    def process_for_net_wo_t(self, t_step=1):
        assert self.train_data is not None, 'Train data is empty.'
        assert self.valid_data is not None, 'Train data is empty.'
        assert self.test_data is not None, 'Train data is empty.'
        assert self.train_data.ndim == 3, 'Train data should be a 3D matrix. Shape: {}'.format(self.train_data.shape)
        assert self.valid_data.ndim == 3, 'Valid data should be a 3D matrix. Shape: {}'.format(self.valid_data.shape)
        assert self.test_data.ndim == 3, 'Test data should be a 3D matrix. Shape: {}'.format(self.test_data.shape)
        self.train_x = np.copy(self.train_data[:, :-t_step, :]).reshape(-1, self.x_points)
        self.train_y = np.copy(self.train_data[:, t_step:, :]).reshape(-1, self.x_points)
        self.valid_x = np.copy(self.valid_data[:, :-t_step, :]).reshape(-1, self.x_points)
        self.valid_y = np.copy(self.valid_data[:, t_step:, :]).reshape(-1, self.x_points)
        self.test_x = np.copy(self.test_data[:, :-t_step, :]).reshape(-1, self.x_points)
        self.test_y = np.copy(self.test_data[:, t_step:, :]).reshape(-1, self.x_points)
        np.testing.assert_array_equal(self.train_x[t_step], self.train_y[0], 'Train_x and _y do not match.')
        np.testing.assert_array_equal(self.valid_x[t_step], self.valid_y[0], 'Valid_x and _y do not match.')
        np.testing.assert_array_equal(self.test_x[t_step], self.test_y[0], 'Test_x and _y do not match.')
        self.train_x = np.expand_dims(self.train_x, axis=-1)
        self.valid_x = np.expand_dims(self.valid_x, axis=-1)
        self.test_x = np.expand_dims(self.test_x, axis=-1)

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
    def get_recur_win_e2e(data, recur_win, t_gap):
        assert data.ndim == 3, 'Input data should be a 3D matrix.'
        d1, d2, d3 = data.shape
        print('d1 {}, d2 {}, d3 {}'.format(d1, d2, d3))
        win_x = np.zeros((d1, d2, d3, 1))
        win_y = np.zeros((d1, d2, d3, recur_win))
        win_id = np.zeros((d1, d2, 2), dtype=int)
        win_dt = np.zeros((d1, d2, recur_win))
        for sample in range(d1):
            for t in range(d2):
                win_x[sample, t, :, 0] = np.copy(data[sample, t, :])
                start = t - (recur_win // 2)
                for i_ in range(recur_win):
                    if start + i_ < 0:
                        win_y[sample, t, :, i_] = np.copy(data[sample, 0, :])
                        win_dt[sample, t, i_] = - t_gap * t
                    elif start + i_ >= d2:
                        win_y[sample, t, :, i_] = np.copy(data[sample, -1, :])
                        win_dt[sample, t, i_] = t_gap * (d2 - 1 - t)
                    else:
                        win_y[sample, t, :, i_] = np.copy(data[sample, start + i_, :])
                        win_dt[sample, t, i_] = t_gap * (i_ - (recur_win // 2))
                win_id[sample, t, :] = [sample, t]
        win_x = win_x.reshape((-1, d3, 1))
        win_y = win_y.reshape((-1, d3, recur_win))
        win_id = win_id.reshape((-1, 2))
        win_dt = win_dt.reshape((-1, 1, recur_win))
        return win_x, win_y, win_id, win_dt

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

    def process_for_recur_net_e2e(self, recur_win=7):
        assert self.train_data is not None, 'Train data is empty.'
        assert self.valid_data is not None, 'Train data is empty.'
        assert self.test_data is not None, 'Train data is empty.'
        assert self.train_data.ndim == 3, 'Train data should be a 3D matrix. Shape: {}'.format(self.train_data.shape)
        assert self.valid_data.ndim == 3, 'Valid data should be a 3D matrix. Shape: {}'.format(self.valid_data.shape)
        assert self.test_data.ndim == 3, 'Test data should be a 3D matrix. Shape: {}'.format(self.test_data.shape)
        assert recur_win % 2 == 1, 'Expecting recur_win to be a odd integer.'
        self.train_x, self.train_y, self.train_id, self.train_dt = self.get_recur_win_e2e(self.train_data,
                                                                                          recur_win=recur_win,
                                                                                          t_gap=self.t_gap)
        print('Shape of train_x: {}, train_y: {}'.format(self.train_x.shape, self.train_y.shape))
        if self.n_valid_sample != 0:
            self.valid_x, self.valid_y, self.valid_id, self.valid_dt = self.get_recur_win_e2e(self.valid_data,
                                                                                              recur_win=recur_win,
                                                                                              t_gap=self.t_gap)
            print('Shape of valid_x: {}, valid_y: {}'.format(self.valid_x.shape, self.valid_y.shape))
        if self.n_test_sample != 0:
            self.test_x, self.test_y, self.test_id, self.test_dt = self.get_recur_win_e2e(self.test_data,
                                                                                          recur_win=recur_win,
                                                                                          t_gap=self.t_gap)
            print('Shape of test_x: {}, test_y: {}'.format(self.test_x.shape, self.test_y.shape))

    def process_for_lsq_w_t(self, sample_id, win=None):
        if not win:
            win_start, win_end = 0, self.train_data.shape[1]
        else:
            win_start, win_end = win[0], win[1]
        dt = PDeGrid.pde_1d_mat(9, 1, win_end - win_start) / self.t_gap
        dt = np.transpose(dt)
        lsq_x = np.copy(self.train_data[sample_id, win_start:win_end, :])
        lsq_y = np.matmul(dt, lsq_x)
        print(self.train_data.shape, lsq_x.shape)
        lsq_t = np.copy(self.time[sample_id, win_start:win_end, 0])
        # mean_time = np.mean(lsq_t)
        # centered_lsq_t = lsq_t - mean_time
        # print(time)
        # print(centered_time)
        # centered_lsq_t = np.expand_dims(centered_lsq_t, axis=1)
        lsq_t -= np.mean(lsq_t)
        lsq_t = np.expand_dims(lsq_t, axis=1)
        return lsq_x, lsq_y, lsq_t

    def process_for_net_w_t(self, sample_id, t_step=1, valid_ratio=0.2, test_ratio=0.2, win=None):
        assert self.train_data is not None, 'Train data is empty.'
        assert self.valid_data is not None, 'Train data is empty.'
        assert self.test_data is not None, 'Train data is empty.'
        assert self.train_data.ndim == 3, 'Train data should be a 3D matrix. Shape: {}'.format(self.train_data.shape)
        assert self.valid_data.ndim == 3, 'Valid data should be a 3D matrix. Shape: {}'.format(self.valid_data.shape)
        assert self.test_data.ndim == 3, 'Test data should be a 3D matrix. Shape: {}'.format(self.test_data.shape)
        if not win:
            win_start, win_end = 0, self.t_points
        else:
            win_start, win_end = win[0], win[1]
        win_wide = win_end - win_start
        valid_cutoff = int((win_end - win_start - t_step) * valid_ratio)
        test_cutoff = int((win_end - win_start - t_step) * test_ratio)
        time = np.copy(self.time[sample_id, win_start: win_end - t_step, 0])
        print('Time Before', time)
        time -= np.mean(time)
        print('Time After', time)
        data = self.train_data[sample_id, win_start: win_end, :]
        data_x = data[:-t_step, :]
        data_y = data[t_step:, :]
        self.train_x = np.copy(data_x[valid_cutoff:win_wide - test_cutoff, :])
        self.train_y = np.copy(data_y[valid_cutoff:win_wide - test_cutoff, :])
        self.train_t = np.copy(time[valid_cutoff:win_wide - test_cutoff])
        self.valid_x = np.copy(data_x[:valid_cutoff, :])
        self.valid_y = np.copy(data_y[:valid_cutoff, :])
        self.valid_t = np.copy(time[:valid_cutoff])
        self.test_x = np.copy(data_x[win_wide - test_cutoff:, :])
        self.test_y = np.copy(data_y[win_wide - test_cutoff:, :])
        self.test_t = np.copy(time[win_wide - test_cutoff:])
        np.testing.assert_array_equal(self.train_x[t_step], self.train_y[0], 'Train_x and _y do not match.')
        if valid_ratio != 0:
            np.testing.assert_array_equal(self.valid_x[t_step], self.valid_y[0], 'Valid_x and _y do not match.')
        if test_ratio != 0:
            np.testing.assert_array_equal(self.test_x[t_step], self.test_y[0], 'Test_x and _y do not match.')
        self.train_t = np.expand_dims(self.train_t, axis=-1)
        self.valid_t = np.expand_dims(self.valid_t, axis=-1)
        self.test_t = np.expand_dims(self.test_t, axis=-1)
        print('~~~~~{} samples in train, {} samples in validation and {} samples in test.'.format(self.train_x.shape[0],
                                                                                                  self.valid_x.shape[0],
                                                                                                  self.test_x.shape[0]))
