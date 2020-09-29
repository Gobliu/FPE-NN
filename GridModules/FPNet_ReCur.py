from keras import layers, models, optimizers, backend
import numpy as np


class FPNetReCur:
    def __init__(self, x_points=None, name=None, recur_win=None, t_gap=None, dx=None, dxx=None):
        assert x_points is not None, 'Please input x_points.'
        assert name is not None, 'Please input name.'
        assert t_gap is not None, 'Please input t_gap.'
        assert recur_win is not None, 'Please input t_step.'
        assert dx is not None, 'Please input dx.'
        assert dxx is not None, 'Please input dxx.'
        self.x_points = x_points
        self.name = name
        self.recur_win = recur_win
        self.t_gap = t_gap
        self.dx = dx
        self.dxx = dxx
        self.dt = np.zeros((1, 1, recur_win))
        self.recur_center = recur_win // 2  # start from 0
        for i in range(recur_win):
            self.dt[0, 0, i] = t_gap * (i - self.recur_center)
        self.dup_x = np.ones((1, 1, recur_win))
        print('dt:', self.dt)

    def recur_train_gh(self, learning_rate, loss):
        x = models.Input(shape=(self.x_points, 1))
        gx = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g')(x)
        gx = layers.Flatten()(gx)
        gx = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gx)
        hx = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h')(x)
        hx = layers.Flatten()(hx)
        hx = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hx)
        dx_dt = layers.Add()([gx, hx])
        dx_dt = layers.Reshape((-1, 1))(dx_dt)
        delta_x = layers.Conv1D(self.recur_win, 1, use_bias=False, name=self.name + 'dt', trainable=False)(dx_dt)
        dup_x = layers.Conv1D(self.recur_win, 1, use_bias=False, name=self.name + 'dup_x', trainable=False)(x)
        answer = layers.Add()([dup_x, delta_x])
        # =============================
        # answer = [dup_x, delta_x, answer1]
        # =============================
        network = models.Model(x, answer)
        network.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss)  # mse: mean squared error
        print(network.summary())
        network.get_layer(name=self.name + 'dx').set_weights([self.dx])  # expecting a list of arrays
        network.get_layer(name=self.name + 'dxx').set_weights([self.dxx])  # expecting a list of arrays
        network.get_layer(name=self.name + 'dt').set_weights([self.dt])
        network.get_layer(name=self.name + 'dup_x').set_weights([self.dup_x])
        return network

    # def recur_gh_x_direct(self, learning_rate, loss):
    #     x = models.Input(shape=(self.x_points, 1))
    #     gx = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g')(x)
    #     gx = layers.Flatten()(gx)
    #     gx = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gx)
    #     hx = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h')(x)
    #     hx = layers.Flatten()(hx)
    #     hx = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hx)
    #     dt_x = layers.Add()([gx, hx])
    #     dt_x = layers.Reshape((-1, 1))(dt_x)
    #     dt_x = layers.Conv1D(self.recur_win, 1, use_bias=False, name=self.name + 'dt', trainable=False)(dt_x)
    #     dup_x = layers.Conv1D(self.recur_win, 1, use_bias=False, name=self.name + 'dup_x', trainable=False)(x)
    #     answer = layers.Add()([dup_x, dt_x])
    #     network = models.Model(x, answer)
    #     network.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss)  # mse: mean squared error
    #     print(network.summary())
    #     network.get_layer(name=self.name + 'dx').set_weights([self.dx])  # expecting a list of arrays
    #     network.get_layer(name=self.name + 'dxx').set_weights([self.dxx])  # expecting a list of arrays
    #     return network

    def recur_train_p_direct(self, learning_rate, loss, fix_g, fix_h):
        x = models.Input(shape=(self.x_points, 1))
        p = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'p')(x)
        gp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g', trainable=False)(p)
        gp = layers.Flatten()(gp)
        gp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gp)
        hp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h', trainable=False)(p)
        hp = layers.Flatten()(hp)
        hp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hp)
        dt_p = layers.Add()([gp, hp])
        dt_p = layers.Reshape((-1, 1))(dt_p)
        delta_p = layers.Conv1D(self.recur_win, 1, use_bias=False, name=self.name + 'dt', trainable=False)(dt_p)
        dup_p = layers.Conv1D(self.recur_win, 1, use_bias=False, name=self.name + 'dup_x', trainable=False)(p)
        answer = layers.Add()([dup_p, delta_p])
        # =============================
        # answer = [delta_p, answer1]
        # =============================
        network = models.Model(x, answer)
        network.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss)  # mse: mean squared error
        print(network.summary())
        network.get_layer(name=self.name + 'dx').set_weights([self.dx])  # expecting a list of arrays
        network.get_layer(name=self.name + 'dxx').set_weights([self.dxx])  # expecting a list of arrays
        network.get_layer(name=self.name + 'dt').set_weights([self.dt])
        network.get_layer(name=self.name + 'dup_x').set_weights([self.dup_x])
        network.get_layer(name=self.name + 'g').set_weights([fix_g])
        network.get_layer(name=self.name + 'h').set_weights([fix_h])
        return network

    def recur_train_p_small_step(self, learning_rate, loss, fix_g, fix_h, small_step=1):
        small_t_gap = self.t_gap / small_step
        small_dt = small_t_gap * np.ones((1, 1, 1))
        one = np.ones((1, 1, 1))

        # backward cell
        p = models.Input(shape=(self.x_points, 1))
        gp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g', trainable=False)(p)
        gp = layers.Flatten()(gp)
        gp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gp)
        hp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h', trainable=False)(p)
        hp = layers.Flatten()(hp)
        hp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hp)
        dt_p = layers.Add()([gp, hp])
        dt_p = layers.Reshape((-1, 1))(dt_p)
        dt_p = layers.Conv1D(1, 1, use_bias=False, name=self.name + 'b_dt', trainable=False)(dt_p)
        next_p = layers.Add()([dt_p, p])
        next_p = layers.Conv1D(1, 1, use_bias=False, name=self.name + 'one', trainable=False, activation='relu')(next_p)
        b_cell = models.Model(p, next_p)
        # print('B Cell Summary')
        # print(b_cell.summary())
        b_cell.get_layer(name=self.name + 'dx').set_weights([self.dx])  # expecting a list of arrays
        b_cell.get_layer(name=self.name + 'dxx').set_weights([self.dxx])
        b_cell.get_layer(name=self.name + 'b_dt').set_weights([-small_dt])  # negative value
        b_cell.get_layer(name=self.name + 'g').set_weights([fix_g])
        b_cell.get_layer(name=self.name + 'h').set_weights([fix_h])
        b_cell.get_layer(name=self.name + 'one').set_weights([one])

        # forward cell
        p = models.Input(shape=(self.x_points, 1))
        gp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g', trainable=False)(p)
        gp = layers.Flatten()(gp)
        gp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gp)
        hp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h', trainable=False)(p)
        hp = layers.Flatten()(hp)
        hp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hp)
        dt_p = layers.Add()([gp, hp])
        dt_p = layers.Reshape((-1, 1))(dt_p)
        dt_p = layers.Conv1D(1, 1, use_bias=False, name=self.name + 'f_dt', trainable=False)(dt_p)
        next_p = layers.Add()([dt_p, p])
        next_p = layers.Conv1D(1, 1, use_bias=False, name=self.name + 'one', trainable=False, activation='relu')(next_p)
        f_cell = models.Model(p, next_p)
        # print('F Cell Summary')
        # print(f_cell.summary())
        f_cell.get_layer(name=self.name + 'dx').set_weights([self.dx])  # expecting a list of arrays
        f_cell.get_layer(name=self.name + 'dxx').set_weights([self.dxx])
        f_cell.get_layer(name=self.name + 'f_dt').set_weights([small_dt])  # negative value
        f_cell.get_layer(name=self.name + 'g').set_weights([fix_g])
        f_cell.get_layer(name=self.name + 'h').set_weights([fix_h])
        f_cell.get_layer(name=self.name + 'one').set_weights([one])

        # RNN
        x = models.Input(shape=(self.x_points, 1))
        p_center = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'p')(x)
        out_p = p_center
        for big_step in range(self.recur_center):
            for step_ in range(small_step):
                if step_ == 0 and big_step == 0:
                    b_next_p = b_cell(p_center)
                else:
                    b_next_p = b_cell(b_cur_p)
                b_cur_p = b_next_p
            out_p = layers.concatenate([b_next_p, out_p])

        for big_step in range(self.recur_center):
            for step_ in range(small_step):
                if step_ == 0 and big_step == 0:
                    f_next_p = f_cell(p_center)
                else:
                    f_next_p = f_cell(f_cur_p)
                f_cur_p = f_next_p
            out_p = layers.concatenate([out_p, f_next_p])
        rnn_model = models.Model(x, out_p)
        rnn_model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss)  # mse: mean squared error
        # print(rnn_model.summary())
        return rnn_model
