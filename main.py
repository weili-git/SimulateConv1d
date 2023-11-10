import torch
import torch.nn as nn
import torch.nn.functional as F

import time


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=4, output_size=1, hidden=2):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class SimulateConv1d(nn.Module):
    def __init__(self, kernel_size, stride, padding, neural_network):
        super(SimulateConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.net = neural_network

    def forward(self, x):   # x: [batch, hidden, time]
        batch_size, hidden_size, input_size = x.shape
        output_size = int((input_size - self.kernel_size + 2 * self.padding) / self.stride) + 1
        x = F.pad(x, (self.padding, self.padding), 'constant', 0)

        windows = torch.zeros(batch_size, hidden_size, output_size, self.kernel_size)

        for i in range(output_size):
            start_idx = i * self.stride
            end_idx = start_idx + self.kernel_size
            windows[:, :, i, :] = x[:, :, start_idx:end_idx]

        windows = windows.view(batch_size*hidden_size, output_size, self.kernel_size)
        results = self.net(windows.float())

        return results.view(batch_size, hidden_size, output_size)


model = SimulateConv1d(4, 2, 1, NeuralNetwork(4, 1, 2))

input_sequence = torch.FloatTensor(64, 100, 20000)

start = time.time()
result = model(input_sequence)
end = time.time()
print("Time elapsed: %.2f" % (end-start))
print(result.shape)
