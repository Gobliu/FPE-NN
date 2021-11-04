from keras import backend as K
import tensorflow as tf


class Loss:
    """Customized loss function"""

    @staticmethod
    def sum_square(y_true, y_pred):
        cost = K.sum(K.square(y_true - y_pred))
        return cost
