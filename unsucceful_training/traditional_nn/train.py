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
#print(x)
#print(y)

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
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
weights_len = len(list(net.parameters()))
input = torch.tensor(x).to(dtype=torch.float32)
# target = F.one_hot(torch.tensor(y).to(dtype=torch.int64), num_classes=3) # cross entropy cannot use one hot
target = torch.tensor(y).to(dtype=torch.int64)

losses = []
accuracies = []

def feed_forward(weights):
    global net, losses, accuracies
    weight_count = 0
    with torch.no_grad():
        for name, param in net.named_parameters(): # update parameters
            #if weight_count == 0: print(param)
            if "weight" in name:
                for i in range(param.data.shape[0]):
                    for j in range(param.data.shape[1]):
                        param.data[i,j] = weights[weight_count]
                        weight_count += 1
            elif "bias" in name:
                param.data *= 0
        # param.data = torch.tensor(weight).to(dtype=torch.float32)
    #print(list(net.parameters()))
    output = net(input)
    output_labels = output.argmax(dim=1)
    ####################################### shape is important here ##########################################
    # output (float32) must be [batch_size, num_classes] (NEVER use argmax -> gradient cannot be calculated)
    # target (int64) must be [batch_size, 1] (dont use one hot)
    ##########################################################################################################
    loss = nn.CrossEntropyLoss()(output, target)
    loss = loss.detach().numpy()
    accuracy = sum(output_labels==target)/(len(output_labels))
    losses.append(float(loss))
    accuracies.append(float(accuracy))
    return loss

weights_len = 0
for name, param in net.named_parameters(): # get number of weights
    param.requires_grad = False
    if "weight" in name:
        for i in range(param.data.shape[0]):
            for j in range(param.data.shape[1]):
                weights_len += 1
initial_weights = [0.5 for i in range(weights_len)]
print(weights_len)

def test(x):
    #print(x)
    loss = sum(x**2)
    losses.append(loss)
    return loss

res = minimize(
    feed_forward, 
    #test, 
    initial_weights,
    method='Nelder-Mead', 
    bounds=[(0,1) for i in range(weights_len)],
    options={"maxiter": 20000},
)

plt.plot(losses)
plt.savefig("loss.png")
plt.clf()
plt.plot(accuracies)
plt.savefig("acc.png")
plt.show()