import numpy as np
from math import exp
from dataclasses import dataclass


np.random.seed(0)

@dataclass
class Config():
    num_hidden_perceptrons: int
    input_size: int


class RecurrentNeuralNetwork(object):
    """
    Architecture is single-hidden-layer
    """

    def __init__(self, config: Config):

        self.config = config

        self.W_xh = np.random.randn(config.num_hidden_perceptrons, config.input_size)
        self.W_hh = np.random.randn(config.num_hidden_perceptrons, config.num_hidden_perceptrons)
        self.W_yh = np.random.randn(config.input_size, config.num_hidden_perceptrons)

        self.b_h = np.zeros((config.num_hidden_perceptrons, 1))
        self.b_y = np.zeros((config.input_size, 1))

        self.history = {}
        self.x = {}
        self.o = {}
        self.q = {}
        self.loss = 0


    def forward_propagation(self, input_index_arr, target_index_arr):
        """

        :param input_index_arr:  The input vector; each element is an index
        :return:
        """

        for t in range(len(input_index_arr)):
            self.x[t] = np.zeros((self.config.input_size, 1))
            self.x[t][input_index_arr[t]] = 1
            self.history[t] = np.tanh(
                np.dot(self.W_hh, self.history[t - 1]) + np.dot(self.W_xh, self.x[t]) + self.b_h
            )
            self.o[t] = np.dot(self.W_yh, self.history[t]) + self.b_y
            self.q[t] = np.exp(self.o[t]) / np.sum(np.exp(self.o[t]))
            self.loss += -np.log(self.q[t][target_index_arr, 0])

    def back_propagation(self, input_index_arr, target_index_arr):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_yh = np.zeros_like(self.W_yh)

        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        for t in reversed(range(len(input_index_arr))):
            partial_loss_over_partial_o = np.copy(self.q[t])
            partial_loss_over_partial_o[target_index_arr[t]] -= 1

            dW_yh += np.dot(partial_loss_over_partial_o, self.history[t].T)
            db_y += partial_loss_over_partial_o

            partial_loss_over_partial_h = np.dot(self.W_yh.T, partial_loss_over_partial_o)