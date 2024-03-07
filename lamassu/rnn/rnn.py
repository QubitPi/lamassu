import numpy as np
from math import exp
from dataclasses import dataclass


np.random.seed(0)

@dataclass
class Config():
    num_hidden_perceptrons: int
    input_size: int
    learning_rate: float


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
        self.b_o = np.zeros((config.input_size, 1))

    def forward_pass(self, input, target, prev_history):
        """

        :param input:  The input vector; each element is an index
        :return:
        """

        history, x, o, q, loss = {}, {}, {}, {}, 0
        history[-1] = np.copy(prev_history)

        for t in range(len(input)):
            x[t] = np.zeros((self.config.input_size, 1))
            x[t][input[t]] = 1

            if t == 0:
                np.dot(self.W_hh, history[t - 1])
                np.dot(self.W_xh, x[t])

            history[t] = np.tanh(
                np.dot(self.W_hh, history[t - 1]) + np.dot(self.W_xh, x[t]) + self.b_h
            )
            o[t] = np.dot(self.W_yh, history[t]) + self.b_o
            q[t] = np.exp(o[t]) / np.sum(np.exp(o[t]))
            loss += -np.log(q[t][target, 0])

        return history, q, x, loss

    def back_propagation(self, input, target, history, q, x):
        gradient_loss_over_W_xh = np.zeros_like(self.W_xh)
        gradient_loss_over_W_hh = np.zeros_like(self.W_hh)
        gradient_loss_over_W_yh = np.zeros_like(self.W_yh)

        gradient_loss_over_b_h = np.zeros_like(self.b_h)
        gradient_loss_over_b_y = np.zeros_like(self.b_o)

        gradient_loss_over_next_h = np.zeros_like(history[0])

        for t in reversed(range(len(input))):
            gradient_loss_over_o = np.copy(q[t])
            gradient_loss_over_o[target[t]] -= 1

            gradient_loss_over_W_yh += np.dot(gradient_loss_over_o, history[t].T)
            gradient_loss_over_b_y += gradient_loss_over_o #

            gradient_loss_over_h = np.dot(self.W_yh.T, gradient_loss_over_o) + gradient_loss_over_next_h
            diag_times_gradient_loss_over_h = (1 - history[t] * history[t]) * gradient_loss_over_h

            gradient_loss_over_b_h += diag_times_gradient_loss_over_h #

            gradient_loss_over_W_xh += np.dot(diag_times_gradient_loss_over_h, x[t].T) #
            gradient_loss_over_W_hh += np.dot(diag_times_gradient_loss_over_h, history[t - 1].T) #

            gradient_loss_over_next_h = np.dot(self.W_hh.T, diag_times_gradient_loss_over_h)

        for gradient in [gradient_loss_over_W_xh, gradient_loss_over_W_hh, gradient_loss_over_W_yh, gradient_loss_over_b_h, gradient_loss_over_b_y]:
            np.clip(gradient, -5, 5, out=gradient) # avoid exploding gradients

        # update weights
        for param, gradient in zip(
                [self.W_xh, self.W_hh, self.W_yh, self.b_h, self.b_o],
                [gradient_loss_over_W_xh, gradient_loss_over_W_hh, gradient_loss_over_W_yh, gradient_loss_over_b_h, gradient_loss_over_b_y]):
            param += -self.config.learning_rate * gradient

        return history[len(input) - 1]

    def inference(self, history, seed_idx):
        x = np.zeros((self.config.input_size, 1))
        x[seed_idx] = 1
        idxes = []

        for timestep in range(200):
            history = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, history) + self.b_h)
            o = np.dot(self.W_yh, history) + self.b_o
            p = np.exp(o) / np.sum(np.exp(o))

            next_idx = self._inference_single(p.ravel())

            x[next_idx] = 1
            idxes.append(next_idx)

        return idxes


    def _inference_single(self, probability_distribution):
        return np.random.choice(range(self.config.input_size), p=probability_distribution)