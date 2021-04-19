from keras import backend as K
import tensorflow as tf


class Loss:
    def __init__(self, pred, true):
        self.pred = pred
        self.true = true
        self.loss_f = None

    @staticmethod
    def sum_square(y_true, y_pred):
        cost = K.sum(K.square(y_true - y_pred))
        return cost

    @staticmethod
    def sum_square_2in1(y_true, y_pred):
        cost = K.sum(K.square(y_true[:, :, :, 0] - y_pred)) + K.sum(K.square(y_true[:, :, :, 1] - y_pred))
        return cost
    # @staticmethod
    # def KL_div(y_true, y_pred):
    #     y_true = tf.clip_by_value(y_true, 1e-15, 1.0)
    #     y_pred = tf.clip_by_value(y_pred, 1e-15, 1.0)
    #     return tf.reduce_sum(y_true * tf.log(y_true / y_pred), axis=1)
    #
    # def JS_div(self, P, Q):
    #     M = (P + Q) / 2.0
    #     l = 0.5 * self.KL_div(P, M) + 0.5 * self.KL_div(Q, M)
    #     return tf.reduce_mean(l) / tf.log(2.0)
