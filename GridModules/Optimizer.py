import tensorflow as tf


class Optimizer:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = None

    def adam(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return self.optimizer
