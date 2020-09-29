import numpy as np
import math
from numpy.linalg import inv


class PartialDerivativeForSN:
    def __init__(self, coord_batch=None, index_batch=None):
        """
        :param coord_batch: the 3D coordinates (2nd column) of n_points in n coordinate systems (1st column).
        :param index_batch: 1 or 0 entries to indicate corresponding point will be used or not in corresponding
        coordinate system.
        :return: two [n, n, 3] matrices to calculate dx and dy of all the used points in each coordinate system.
        """
        assert coord_batch.ndim == 3, 'Please input coord_batch as 3D matrix [n, n, 3].'
        assert index_batch.ndim == 3, 'Please input index_batch as 3D matrix [n, n, 1].'
        self.n_points = coord_batch.shape[0]
        assert self.n_points == coord_batch.shape[1], 'Shape of coord_batch is incorrect. {}'.format(coord_batch.shape)
        assert self.n_points == index_batch.shape[0], 'Shape of index_batch is incorrect. {}'.format(index_batch.shape)
        assert self.n_points == index_batch.shape[1], 'Shape of index_batch is incorrect. {}'.format(index_batch.shape)
        self.coord_batch = coord_batch
        self.index_batch = index_batch

    @staticmethod
    def get_t_sro_2d(neighbour_count):
        t_sro = 0
        usable_points = t_sro + 1  # max points could be used to improve accuracy.
        while usable_points < neighbour_count:
            t_sro += 1
            usable_points += t_sro + 1
        usable_points -= t_sro + 1
        t_sro -= 1
        print('Neighbour_points: {} \t t_sro: {}'.format(neighbour_count, t_sro))
        return t_sro

    @staticmethod
    def get_rank_list_use_all(neighbour_count, t_sro):
        rank_list = []
        dx_rank = 0
        while dx_rank <= t_sro:
            dy_rank = 0
            while dx_rank + dy_rank <= t_sro:
                rank_list.append([dx_rank, dy_rank])
                dy_rank += 1
            dx_rank += 1
        count = len(rank_list)
        dx_rank = 0
        dy_rank = t_sro + 1
        while count < neighbour_count:
            rank_list.append([dx_rank, dy_rank])
            dx_rank += 1
            dy_rank -= 1
            count += 1
        print('Rank List:', rank_list)
        assert len(rank_list) == neighbour_count, 'Rank List length {} does not match the neighbour count {}.'.format(
            len(rank_list), neighbour_count)
        return rank_list

    def get_coord(self, p_index):
        usable_points_index = self.index_batch[p_index, :, 0] == 1
        usable_points_coord = self.coord_batch[p_index, usable_points_index, :]
        return usable_points_index, usable_points_coord

    @staticmethod
    def get_neighbour_filter_mat_2d(self, usable_points_coord, rank_list, dxdy):
        mat_a = np.zeros((usable_points_coord.shape[0], usable_points_coord.shape[0]))
        assert dxdy in rank_list, 'Target derivative {} not in rank_list {}.'.format(dxdy, rank_list)
        rank_value = np.zeros((len(rank_list), 1))
        rank_value[rank_list.index(dxdy)] = 1
        for i in range(usable_points_coord):
            dx_rank, dy_rank = rank_list[i]
            co_x = usable_points_coord[:, 0] ** dx_rank
            co_y = usable_points_coord[:, 1] ** dy_rank
            co_ = 1 / math.factorial(dx_rank + dy_rank)
            mat_a[i] = co_x * co_y * co_
        neighbour_filter_mat = np.matmul(inv(mat_a), rank_value)
        return neighbour_filter_mat

    def get_total_filter_mat_use_all(self, dxdy):
        total_filter_mat = np.zeros((self.n_points, self.n_points))
        for p in range(self.n_points):
            neighbour_count = np.sum(self.index_batch(p))
            t_sro = self.get_t_sro_2d(neighbour_count)
            rank_list = self.get_rank_list_use_all(neighbour_count, t_sro)
            usable_points_index, usable_points_coord = self.get_coord(p)
            neighbour_filter_mat = self.get_neighbour_filter_mat_2d(usable_points_coord, rank_list, dxdydz=dxdy)
            total_filter_mat[p, usable_points_index] = neighbour_filter_mat[:, 0]
        return total_filter_mat

    def cal_total_filter_mat_for_sn(self):
        total_filter_mat_dx = self.get_total_filter_mat_use_all(dxdy=[1, 0])
        total_filter_mat_dy = self.get_total_filter_mat_use_all(dxdy=[0, 1])
        return total_filter_mat_dx, total_filter_mat_dy
