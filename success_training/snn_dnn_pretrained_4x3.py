from contextlib import redirect_stdout

# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import snntorch.functional as SF
#loss_fn = SF.ce_count_loss()
loss_fn = SF.ce_rate_loss()
acc_fn = SF.acc.accuracy_rate

import torch
import torch.nn as nn
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
import pygad
from sklearn.datasets import fetch_covtype, load_iris
from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()
#import datetime
#dt = datetime.datetime.now()

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
    ax[2].plot(spk)
    plt.ylabel("Output spikes")
    plt.yticks([]) 

    plt.show()

#with open(f'out_{datetime.datetime.timestamp(dt)}.txt', 'w') as f:
#    with redirect_stdout(f):
num_steps = 200

dataset = load_iris()
#dataset = fetch_covtype()
x = dataset['data']
x = mm.fit_transform(x)
#plt.hist(x)
#plt.show()
y = dataset['target']
#print(x)
#print(y)

# layer parameters
num_inputs = 4
num_outputs = 3
beta = 1

# initialize layers
fc1 = nn.Linear(num_inputs, num_outputs).to(device)
lif1 = snn.Leaky(beta=beta, reset_mechanism = "zero").to(device)
for layer in [fc1, lif1]:
    for param in layer.parameters():
        param.requires_grad = False
weights_len = num_inputs*num_outputs

target = torch.tensor(y).to(dtype=torch.int64).to(device)

losses = []
accuracies = []

spk_in = spikegen.rate_conv(torch.tensor([x for i in range(num_steps)], dtype=torch.float32)).to(device)

def feed_forward(weights):
    #print(weights)
    global losses, accuracies, mem1, mem2
    # update weights    
    weight_count = 0
    with torch.no_grad():
        for layer in [fc1]:
            for name, param in layer.named_parameters(): # update parameters
                #if weight_count == 0: print(param)
                if "weight" in name:
                    for i in range(param.data.shape[0]):
                        for j in range(param.data.shape[1]):
                            param.data[i,j] = weights[weight_count]
                            weight_count += 1
                if "bias" in name:
                    param.data *= 0

    # feed forward through batches
    outputs = []
    #print(f"{spk_in.shape}")
    #print(f"Dimensions of spk_in: {spk_in.size()}")

    # Initialize hidden states
    mem1 = lif1.init_leaky()

    # record outputs
    cur1_rec = []
    mem1_rec = []
    spk1_rec = []

    # network simulation
    for step in range(num_steps):
        cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
        spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
        cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
        spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane

        cur1_rec.append(cur1)
        mem1_rec.append(mem1)
        spk1_rec.append(spk1)

    # convert lists to tensors
    cur1_rec = torch.stack(cur1_rec)
    mem1_rec = torch.stack(mem1_rec)
    spk1_rec = torch.stack(spk1_rec)

    #plot_cur_mem_spk(
    #    cur1_rec[:,0,0].detach().numpy(), 
    #    mem1_rec[:,0,0].detach().numpy(), 
    #    spk1_rec[:,0,0].detach().numpy(), 
    #)

    loss = float(loss_fn(spk1_rec, target))
    accuracy = float(acc_fn(spk1_rec, target))
    print(f"loss={loss:.7f}, accuracy={accuracy:.7f}")
    # print(np.max(outputs)) # 168
    return 1/loss

fitness_function = feed_forward

num_generations = 100
num_parents_mating = 8

sol_per_pop = 32
num_genes = weights_len

init_range_low = 0
init_range_high = 1

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 50

feed_forward([5.26298039e-04, 1.00000000e+00, 2.87677457e-04, 0.00000000e+00,
       0.00000000e+00, 1.79793856e-05, 9.99853842e-01, 9.99939527e-01,
       7.04145093e-01, 6.06967556e-05, 9.99979508e-01, 1.00000000e+00])

plt.plot(losses)
plt.savefig("snn_loss.png")
plt.clf()
plt.plot(accuracies)
plt.savefig("snn_acc.png")
plt.show()
