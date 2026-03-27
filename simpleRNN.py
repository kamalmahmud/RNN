import numpy as np


class SimpleRNN:
    def __init__(self, input_size=9, hidden_size=4, output_size=3):
        self.W_xh = np.random.rand(hidden_size, input_size)  # input to hidden 4x9
        self.W_ho = np.random.rand(output_size, hidden_size)  # hidden to output 3x4
        self.b_xh = np.zeros((hidden_size, 1))  # hidden bias
        self.b_ho = np.zeros((output_size, 1))  # output bias

    def forward(self, inputs):
        steps_outputs, hidden_states = {}, {}
        hidden_states[-1] = np.zeros((self.W_xh.shape[0], 1))  # no history for idx -1

        for i in range(len(inputs)):
            x = np.array(inputs[i]).reshape(-1, 1)  #9x1
            cur_hidden = np.dot(self.W_xh, x) + self.b_xh  # 4x9 * 9x1 + 4x1 = 4x1

            hidden_states[i] = hidden_states[i - 1]
            hidden_states[i] += cur_hidden  # adding the previous state to the current
            hidden_states[i] = np.tanh(hidden_states[i])  # non-linear transformation

            steps_outputs[i] = np.dot(self.W_ho, hidden_states[i]) + self.b_ho

        return steps_outputs, hidden_states


x = np.random.rand(4, 9).shape[0]
print(x)
print(
    np.zeros((x, 1))
)
