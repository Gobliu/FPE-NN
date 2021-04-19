from keras import layers, models, optimizers, backend
from .PDM_NG import PDM_NG

import numpy as np


class FPENet_NG:
    def __init__(self, x_coord=None, name=None, t_sro=None):
        assert x_coord is not None, 'Please input x_points.'
        assert name is not None, 'Please input name.'
        assert t_sro is not None, 'Please input t_sro.'
        self.x_points = x_coord.shape[0]
        self.name = name
        self.dx = PDM_NG.pde_1d_mat(x_coord, t_sro, sro=1)
        self.dxx = PDM_NG.pde_1d_mat(x_coord, t_sro, sro=2)

    def recur_train_gh(self, learning_rate, loss):
        x = models.Input(shape=(self.x_points, 1))
        t = models.Input(shape=(1, None))
        gx = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g')(x)
        gx = layers.Flatten()(gx)
        gx = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gx)
        hx = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h')(x)
        hx = layers.Flatten()(hx)
        hx = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hx)
        dx_dt = layers.Add()([gx, hx])
        dx_dt = layers.Reshape((-1, 1))(dx_dt)              # shape [batch, x_points, 1]
        delta_x = layers.Dot(axes=(2, 1))([dx_dt, t])         # shape [batch, x_points, t_size]
        answer = layers.Add()([x, delta_x])                 # might got problem
        # ===================
        # answer = [x, delta_x, answer1]
        # ===================
        network = models.Model([x, t], answer)
        network.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss)     # mse: mean squared error
        print(network.summary())
        network.get_layer(name=self.name + 'dx').set_weights([self.dx])             # expecting a list of arrays
        network.get_layer(name=self.name + 'dxx').set_weights([self.dxx])           # expecting a list of arrays
        return network

    def recur_train_p_direct(self, learning_rate, loss, fix_g, fix_h):
        x = models.Input(shape=(self.x_points, 1))
        t = models.Input(shape=(1, None))
        p = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'p')(x)
        gp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g', trainable=False)(p)
        gp = layers.Flatten()(gp)
        gp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gp)
        hp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h', trainable=False)(p)
        hp = layers.Flatten()(hp)
        hp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hp)
        dp_dt = layers.Add()([gp, hp])
        dp_dt = layers.Reshape((-1, 1))(dp_dt)      # shape [batch, x_points, 1]
        delta_p = layers.Dot(axes=(2, 1))([dp_dt, t])  # shape [batch, x_points, t_size]
        answer = layers.Add()([p, delta_p])           # might got problem
        # ===================
        # answer = [delta_p, answer1]
        # ===================
        network = models.Model([x, t], answer)
        network.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss)  # mse: mean squared error
        print(network.summary())
        network.get_layer(name=self.name + 'dx').set_weights([self.dx])  # expecting a list of arrays
        network.get_layer(name=self.name + 'dxx').set_weights([self.dxx])  # expecting a list of arrays
        network.get_layer(name=self.name + 'g').set_weights([fix_g])
        network.get_layer(name=self.name + 'h').set_weights([fix_h])
        return network

    def recur_train_3in1(self, learning_rate, loss):
        x = models.Input(shape=(self.x_points, 1))
        t = models.Input(shape=(1, None))
        p = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'p')(x)
        gp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'g')(p)
        gp = layers.Flatten()(gp)
        gp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dx', trainable=False)(gp)
        hp = layers.LocallyConnected1D(1, 1, use_bias=False, name=self.name + 'h')(p)
        hp = layers.Flatten()(hp)
        hp = layers.Dense(self.x_points, use_bias=False, name=self.name + 'dxx', trainable=False)(hp)
        dp_dt = layers.Add()([gp, hp])
        dp_dt = layers.Reshape((-1, 1))(dp_dt)      # shape [batch, x_points, 1]
        delta_p = layers.Dot(axes=(2, 1))([dp_dt, t])  # shape [batch, x_points, t_size]
        answer = layers.Add()([p, delta_p])           # might got problem
        # ===================
        # answer = [delta_p, answer1]
        # ===================
        network = models.Model([x, t], answer)
        network.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss)  # mse: mean squared error
        print(network.summary())
        network.get_layer(name=self.name + 'dx').set_weights([self.dx])  # expecting a list of arrays
        network.get_layer(name=self.name + 'dxx').set_weights([self.dxx])  # expecting a list of arrays
        # network.get_layer(name=self.name + 'g').set_weights([fix_g])
        # network.get_layer(name=self.name + 'h').set_weights([fix_h])
        return network
