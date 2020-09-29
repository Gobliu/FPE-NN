import numpy as np
import math
from numpy.linalg import inv


class PartialDerivative:
    def __init__(self, coord=None, cutoff=None):
        assert coord is not None, 'Please input coord as 2D matrix, [?,1], [?,2] or [?,3].'
        assert coord.ndim == 2, 'Please input coord as 2D matrix, [?,1], [?,2] or [?,3].'
        assert cutoff is not None, 'Please insert a cutoff distance.'
        self.coord = coord
        self.n_points = self.coord.shape[0]
        self.dimension = self.coord.shape[1]
        assert self.dimension <= 3, 'Dimension should not be larger than 3.'
        self.cutoff = cutoff

    def get_neighbour_count(self, shift_coord):
        dist = np.sqrt(np.sum(shift_coord**2, axis=1))  # shape: (n_points,)
        neighbour_count = (dist < self.cutoff).sum()
        return neighbour_count

    def get_t_sro(self, neighbour_count):
        t_sro = 0
        usable_points = 0
        if self.dimension == 1:
            usable_points = neighbour_count  # max points could be used to improve accuracy.
            t_sro = neighbour_count
        elif self.dimension == 2:
            usable_points = t_sro + 1  # max points could be used to improve accuracy.
            while usable_points < neighbour_count:
                t_sro += 1
                usable_points += t_sro + 1
            usable_points -= t_sro + 1
            t_sro -= 1
        elif self.dimension == 3:
            sro_type = t_sro
            usable_points = sro_type  # max points could be used to improve accuracy.
            while usable_points <= neighbour_count:
                t_sro += 1
                sro_type = 0
                for i in range(t_sro+1):
                    sro_type += i + 1
                usable_points += sro_type
            usable_points -= sro_type
            t_sro -= 1
        print('Dimension: {} \t Neighbour_points: {} \t Usable_points: {} \t t_sro: {}'.format(self.dimension,
                                                                                               neighbour_count,
                                                                                               usable_points,
                                                                                               t_sro))
        return usable_points, t_sro

    def get_rank_list(self, t_sro):
        rank_list = []
        dx_rank = 0
        if self.dimension == 1:
            while dx_rank <= t_sro:
                rank_list.append([dx_rank])
                dx_rank += 1
        elif self.dimension == 2:
            while dx_rank <= t_sro:
                dy_rank = 0
                while dx_rank + dy_rank <= t_sro:
                    rank_list.append([dx_rank, dy_rank])
                    dy_rank += 1
                dx_rank += 1
        elif self.dimension == 3:
            while dx_rank <= t_sro:
                dy_rank = 0
                while dx_rank + dy_rank <= t_sro:
                    dz_rank = 0
                    while dx_rank + dy_rank + dz_rank <= t_sro:
                        rank_list.append([dx_rank, dy_rank, dz_rank])
                        dz_rank += 1
                    dy_rank += 1
                dx_rank += 1
        print(rank_list)
        return rank_list

    @staticmethod
    def get_neighbour_index(shift_coord, usable_points):
        dist = np.sqrt(np.sum(shift_coord ** 2, axis=1))  # shape: (n_points,)
        usable_points_index = np.argsort(dist)[:usable_points]  # shape: (usable_points,)
        usable_points_coord = shift_coord[usable_points_index]
        return usable_points_index, usable_points_coord

    def get_neighbour_filter_mat(self, usable_points_coord, rank_list, dxdydz):
        mat_a = np.zeros((usable_points_coord.shape[0], usable_points_coord.shape[0]))
        assert dxdydz in rank_list, 'Target derivative {} not in rank_list {}.'.format(dxdydz, rank_list)
        rank_value = np.zeros((len(rank_list), 1))
        rank_value[rank_list.index(dxdydz)] = 1
        if self.dimension == 1:
            for i in range(usable_points_coord):
                dx_rank = rank_list[i]
                co_x = usable_points_coord[:, 0] ** dx_rank
                co_ = 1 / math.factorial(dx_rank)
                mat_a[i] = co_x * co_
        elif self.dimension == 2:
            for i in range(usable_points_coord):
                dx_rank, dy_rank = rank_list[i]
                co_x = usable_points_coord[:, 0] ** dx_rank
                co_y = usable_points_coord[:, 1] ** dy_rank
                co_ = 1 / math.factorial(dx_rank + dy_rank)
                mat_a[i] = co_x * co_y * co_
        elif self.dimension == 3:
            for i in range(usable_points_coord):
                dx_rank, dy_rank, dz_rank = rank_list[i]
                co_x = usable_points_coord[:, 0] ** dx_rank
                co_y = usable_points_coord[:, 1] ** dy_rank
                co_z = usable_points_coord[:, 2] ** dz_rank
                co_ = 1 / math.factorial(dx_rank + dy_rank + dz_rank)
                mat_a[i] = co_x * co_y * co_z * co_
        neighbour_filter_mat = np.matmul(inv(mat_a), rank_value)
        return neighbour_filter_mat

    def get_total_filter_mat_minimum_points(self, dxdydz):
        total_filter_mat = np.zeros((self.n_points, self.n_points))
        for p in range(self.n_points):
            shifted_coord = self.coord - self.coord[p]
            neighbour_count = self.get_neighbour_count(shifted_coord)
            usable_points, t_sro = self.get_t_sro(neighbour_count)
            rank_list = self.get_rank_list(t_sro)
            usable_points_index, usable_points_coord = self.get_neighbour_index(shifted_coord, usable_points)
            neighbour_filter_mat = self.get_neighbour_filter_mat(usable_points_coord, rank_list, dxdydz=dxdydz)
            total_filter_mat[p, usable_points_index] = neighbour_filter_mat[:, 0]

    def cal_total_filter_mat_for_sn(self, dxdy):
        total_filter_mat = np.zeros((self.n_points, self.n_points))
        for p in range(self.n_points):
            shifted_coord = self.coord - self.coord[p]
            neighbour_count = self.get_neighbour_count(shifted_coord)

    # @staticmethod
    # def pde_2d_coord(value, coord, dxdy, accuracy=False):
    #     """
    #     input: points with value, 2D coordinate, desire derivative.
    #     Optional: Report accuracy or not (default is False)
    #     return: derivative value at the first input point is the point to calculate derivative.
    #     """
    #     assert value.ndim == 2, 'Input value as a 2D matrix with shape [?, 1].'
    #     assert coord.ndim == 2, 'Input coord(coordinate) as a 2D matrix and x coordinates in first column'
    #     assert len(dxdy) == 2, 'Input dxdy as list with two integers.'
    #     assert type(dxdy[0]) == int, 'dxdy[0] should be integer.'
    #     assert type(dxdy[1]) == int, 'dxdy[1] should be integer.'
    #     assert value.shape[0] == coord.shape[0], 'Shape of value and coord do not match.'
    #     sro = sum(dxdy)
    #     points = value.shape[0]
    #
    #     new_coord = coord - coord[0]        # ensure the target point at origin
    #     ######### test
    #     sro = 2
    #     points = 10
    #     dxdy = [1, 1]
    #     #########
    #
    #     min_points = 0
    #     for i in range(sro+1):
    #         min_points += i + 1
    #     assert min_points <= points, 'Not enough points to calculate desire derivative.' \
    #                                  ' At least {} points.'.format(min_points)
    #
    #     t_sro = 0
    #     max_points = t_sro + 1      # max_points is the max points could be used to improve accuracy.
    #     while max_points + t_sro + 1 < points:
    #         t_sro += 1
    #         max_points += t_sro + 1
    #
    #     if accuracy:
    #         print('Total sum of rule is {}, sum of rule is {} and accuracy is O(e^{}).'.format(t_sro, sro, t_sro - sro))
    #
    #     rank_list = []
    #     dx_rank = 0
    #     while dx_rank <= t_sro:
    #         dy_rank = 0
    #         while dx_rank + dy_rank <= t_sro:
    #             rank_list.append([dx_rank, dy_rank])
    #             dy_rank += 1
    #         dx_rank += 1
    #     print(rank_list)
    #     rank_value = np.zeros((max_points, 1))
    #     rank_value[rank_list.index(dxdy)] = 1
    #     print(rank_value)
    #
    #     mat_a = np.zeros((max_points, points))
    #     test_mat_a = np.zeros((max_points, points))
    #     for i in range(max_points):
    #         dx_rank, dy_rank = rank_list[i]
    #         co_x = new_coord[:, 0] ** dx_rank
    #         co_y = new_coord[:, 1] ** dy_rank
    #         co_ = 1 / math.factorial(dx_rank + dy_rank)
    #         test_mat_a[i] = co_x * co_y * co_
    #         # the same as above
    #         mat_a[i] = new_coord[:, 0] ** rank_list[i][0] * new_coord[:, 1] ** rank_list[i][1] / \
    #             math.factorial(sum(rank_list[i]))
    #     np.testing.assert_array_equal(mat_a, test_mat_a)
    #
    #     # use linear least square to use all the given points
    #     co_mat = np.matmul(np.matmul(inv(np.matmul(mat_a.transpose(), mat_a)), mat_a.transpose()), rank_value)
    #     answer = np.sum(co_mat * value)
    #     return answer
    #
    # @staticmethod
    # def pde_3d_mat(coord, cutoff, dxdydz):
    #     """
    #     input: coord, 3D coordinate of all the points; cutoff, a value only the points whose distance to target point
    #         shorter than cutoff are used to calculate gradient. dxdydz, a list with three integer element indicating the
    #         derivative rank.
    #     return: three matrices to calculate gradient.
    #     """
    #     assert coord.ndim == 2, 'Input coordinate should be [?, 2] shape.'
    #     points = coord.shape[0]
    #     sro = sum(dxdydz)
    #
    #     answer = np.zeros((points, points))
    #     for p in range(points):
    #         new_coord = coord - coord[p]
    #         dist = np.sqrt(np.sum(new_coord**2, axis=1))
    #         index = np.nonzero(dist < cutoff)[0]
    #         neighbour = new_coord[dist < cutoff]
    #         n_points = len(index)
    #         print(p, neighbour.shape)
    #
    #         ### test
    #         # sro = 3
    #         # dxdydz = [3, 0, 0]
    #         # n_points = 20
    #         ### test
    #
    #         t_sro = 0
    #         max_points = t_sro + 1  # max_points is the max points could be used to improve accuracy.
    #         while max_points <= n_points:
    #             t_sro += 1
    #             sro_type = 0
    #             for i in range(t_sro+1):
    #                 sro_type += i + 1
    #             max_points += sro_type
    #         max_points -= sro_type
    #         t_sro -= 1
    #         # print(n_points, max_points, t_sro)
    #
    #         assert t_sro >= sro, 'Not enough points to calculate derivative.'
    #         rank_list = []
    #         dx_rank = 0
    #         while dx_rank <= t_sro:
    #             dy_rank = 0
    #             while dx_rank + dy_rank <= t_sro:
    #                 dz_rank = 0
    #                 while dx_rank + dy_rank + dz_rank <= t_sro:
    #                     rank_list.append([dx_rank, dy_rank, dz_rank])
    #                     dz_rank += 1
    #                 dy_rank += 1
    #             dx_rank += 1
    #         # print(rank_list)
    #         rank_value = np.zeros((max_points, 1))
    #         rank_value[rank_list.index(dxdydz)] = 1
    #         # print(rank_value.transpose())
    #
    #         mat_a = np.zeros((max_points, n_points))
    #         print('neighbour coord')
    #         print(neighbour[:, 0])
    #         for i in range(max_points):
    #             dx_rank, dy_rank, dz_rank = rank_list[i]
    #             co_x = neighbour[:, 0] ** dx_rank
    #             co_y = neighbour[:, 1] ** dy_rank
    #             co_z = neighbour[:, 2] ** dz_rank
    #             co_ = 1 / math.factorial(dx_rank + dy_rank + dz_rank)
    #             mat_a[i] = co_x * co_y * co_z * co_
    #             # print(co_)
    #             # print(rank_list[i])
    #         # print(np.linalg.det(mat_a))
    #         # use linear least square to use all the given points
    #         # print(np.matmul(mat_a.transpose(), mat_a))
    #         co_mat = np.matmul(np.matmul(inv(np.matmul(mat_a.transpose(), mat_a)), mat_a.transpose()), rank_value)
    #         # for p_ in range(n_points):
    #         #     answer[p, index[p_]] = co_mat[p_]
    #         answer[p, index] = co_mat[:, 0]
    #     return answer
