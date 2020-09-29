from keras import backend as K


class Loss:
    def __init__(self, pred, true):
        self.pred = pred
        self.true = true
        self.loss_f = None

    @staticmethod
    def sum_square(y_true, y_pred):
        cost = K.sum(K.square(y_true - y_pred))
        return cost
