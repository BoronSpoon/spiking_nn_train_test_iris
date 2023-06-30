# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
from scipy.optimize import minimize
from sklearn.datasets import load_iris

num_steps = 200

iris = load_iris()
x = iris['data']
y = iris['target']
#print(x)
#print(y)

#@title Plotting Settings
def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, 
                            gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([0, ylim_max1])
    ax[0].set_xlim([0, 200])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    if title:
        ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([0, ylim_max2]) 
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    plt.ylabel("Output spikes")
    plt.yticks([]) 

    plt.show()

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,7), sharex=True, 
                            gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input spikes
    splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s = 0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([-0.2, 2.2])

    plt.show()

# layer parameters
num_inputs = 4
num_hidden = 10
num_outputs = 3
beta = 1

# initialize layers
fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Leaky(beta=beta)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta)
for layer in [fc1, lif1, fc2, lif2]:
    for param in layer.parameters():
        param.requires_grad = False
weights_len = num_inputs*num_hidden + num_hidden*num_outputs

# Initialize hidden states
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

target = torch.tensor(y).to(dtype=torch.int64)

losses = []
accuracies = []

def feed_forward(weights):
    global losses, accuracies, mem1, mem2
    # update weights    
    weight_count = 0
    with torch.no_grad():
        for layer in [fc1, fc2]:
            for name, param in layer.named_parameters(): # update parameters
                #if weight_count == 0: print(param)
                if "weight" in name:
                    for i in range(param.data.shape[0]):
                        for j in range(param.data.shape[1]):
                            param.data[i,j] = weights[weight_count]
                            weight_count += 1

    # feed forward through batches
    outputs = []
    for batch_count in range(150):
        #if batch_count%10 == 0: print(batch_count)
        spk_in = spikegen.rate_conv(torch.tensor([x[batch_count] for i in range(num_steps)], dtype=torch.float32)).unsqueeze(1)
        #print(f"{spk_in.shape}")
        #print(f"Dimensions of spk_in: {spk_in.size()}")

        # record outputs
        mem2_rec = []
        spk1_rec = []
        spk2_rec = []

        # network simulation
        for step in range(num_steps):
            cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
            spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
            cur2 = fc2(spk1)
            spk2, mem2 = lif2(cur2, mem2)

            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        # convert lists to tensors
        mem2_rec = torch.stack(mem2_rec)
        spk1_rec = torch.stack(spk1_rec)
        spk2_rec = torch.stack(spk2_rec)

        output = spk2_rec.detach().numpy().sum(axis=0)[0]/170
        outputs.append(output)
    
    outputs = torch.tensor(outputs, dtype=torch.float32)
    output_labels = outputs.argmax(dim=1)
    ####################################### shape is important here ##########################################
    # output (float32) must be [batch_size, num_classes] (NEVER use argmax -> gradient cannot be calculated)
    # target (int64) must be [batch_size, 1] (dont use one hot)
    ##########################################################################################################
    loss = nn.CrossEntropyLoss()(outputs, target)
    loss = loss.detach().numpy()
    accuracy = sum(output_labels==target)/(len(output_labels))
    print(loss, accuracy)
    losses.append(float(loss))
    accuracies.append(float(accuracy))
    # print(np.max(outputs)) # 168
    return loss

initial_weights = [0 for i in range(weights_len)]
    
res = minimize(
    feed_forward, 
    #test, 
    initial_weights,
    method='Nelder-Mead', 
    bounds=[(0,10) for i in range(weights_len)],
    options={"maxiter": 100},
)

plt.plot(losses)
plt.savefig("snn_loss.png")
plt.clf()
plt.plot(accuracies)
plt.savefig("snn_acc.png")
plt.show()
