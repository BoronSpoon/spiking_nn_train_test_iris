import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import load_iris
torch.manual_seed(0)

iris = load_iris()
x = iris['data']
y = iris['target']
print(x)
print(y)

plot_params = {
    "axes.labelsize": 16,
    "axes.titlesize": 16,    
    "lines.linestyle": "solid",
    "lines.linewidth": 1,
    "lines.marker": "o",
    "lines.markersize": 3,
    "xtick.major.size": 3.5,
    "xtick.minor.size": 2,
    "xtick.labelsize": 13,
    "ytick.major.size": 3.5,
    "ytick.minor.size": 2,
    "ytick.labelsize": 13,
}
plt.rcParams.update(plot_params)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
weights_len = len(list(net.parameters()))
def feed_forward(weights):
    global net
    for param, weight in zip(net.parameters(), weights): # update parameters
        param = torch.tensor(weight).to(dtype=torch.float32)
    input = torch.tensor(x).to(dtype=torch.float32)
    output = net(input)
    target = F.one_hot(torch.tensor(y), num_classes=3).to(dtype=torch.float32)  # a dummy target, for example
    loss = nn.CrossEntropyLoss()(output, target)
    return loss

initial_weights = [1 for i in range(weights_len)]
res = minimize(
    feed_forward, 
    initial_weights,
    method='BFGS', 
    bounds=[(0,1) for i in range(weights_len)],
)