import numpy as np

from lamassu.rnn.rnn import Config
from lamassu.rnn.rnn import RecurrentNeuralNetwork

if __name__ == "__main__":
    num_hidden_perceptrons= 100
    seq_length = 25
    learning_rate = 1e-1


    data = open('pride-and-prejudice.txt', 'r').read()
    char_set = list(set(data))
    num_chars, num_unique_chars = len(data), len(char_set)
    char_to_idx = { ch:i for i,ch in enumerate(char_set) }
    idx_to_char = { i:ch for i,ch in enumerate(char_set) }

    rnn = RecurrentNeuralNetwork(
        Config(
            num_hidden_perceptrons=num_hidden_perceptrons,
            input_size=num_unique_chars,
            learning_rate=learning_rate
        )
    )

    num_iter, pointer = 0, 0


    while True:
        if pointer + seq_length + 1 >= len(data) or num_iter == 0:
            prev_history = np.zeros((num_hidden_perceptrons, 1))
            pointer = 0
        input = [char_to_idx[c] for c in data[pointer: pointer + seq_length]]
        target = [char_to_idx[c] for c in data[pointer + 1: pointer + seq_length + 1]]

        if num_iter % 100 == 0: # inference after every 100 trainings
            inferenced_idxes = rnn.inference(prev_history, input[0])
            inferenced = ''.join(idx_to_char[idx] for idx in inferenced_idxes)
            print("============ inference ============")
            print(inferenced)

        history, q, x, loss = rnn.forward_pass(input, target, prev_history)

        if num_iter % 100 == 0:
            print("loss: {}".format(loss))

        prev_history = rnn.back_propagation(input, target, history, q, x)

        pointer += seq_length
        num_iter += 1