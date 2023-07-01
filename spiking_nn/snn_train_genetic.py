# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
import pygad
from sklearn.datasets import load_iris

num_steps = 200

iris = load_iris()
x = iris['data']
x = x/np.max(x)
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
lif1 = snn.Leaky(beta=beta, reset_mechanism = "zero", threshold = 4) # 4 incoming neurons (threshold = 4)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta, reset_mechanism = "zero", threshold = 10) # 10 incoming neurons (threshold = 10)
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

spk_ins = [spikegen.rate_conv(torch.tensor([x[batch_count] for i in range(num_steps)], dtype=torch.float32)).unsqueeze(1) for batch_count in range(150)]

def feed_forward(ga_instance, weights, solution_idx):
    print(weights[:20])
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
                if "bias" in name:
                    param.data *= 0

    # feed forward through batches
    outputs = []
    for batch_count in range(150):
        #if batch_count%10 == 0: print(batch_count)
        spk_in = spk_ins[batch_count]
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

        output = spk2_rec.detach().numpy().sum(axis=0)[0]/200
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
    #print(outputs[:5, :3])
    print(f"loss={loss:.3f}, accuracy={accuracy:.3f}")
    losses.append(float(loss))
    accuracies.append(float(accuracy))
    # print(np.max(outputs)) # 168
    return 1/loss

#initial_weights = [0.05 for i in range(weights_len)]

# training result of normal nn
initial_weights = [
    0.84783617, 0.44503967, 0.56239707, 1.        , 0.42604591,
    0.01138287, 0.96939782, 0.97666726, 0.00675524, 0.        ,
    0.88647085, 0.20231477, 0.00146853, 0.1888283 , 0.25861927,
    0.99776879, 0.1054487 , 0.        , 0.49059965, 0.98568222,
    0.99958481, 0.05757991, 0.02662554, 0.0178756 , 0.4848848 ,
    0.44408592, 0.10377315, 0.24340086, 0.04875144, 0.06642141,
    0.14724149, 0.96571841, 0.1582396 , 0.98016019, 0.01885701,
    1.        , 0.95399136, 0.        , 0.48079907, 0.93373499,
    0.3635811 , 0.00378183, 0.05886937, 0.38260926, 0.38000544,
    0.52873809, 0.99886486, 0.26816627, 1.        , 0.38996802,
    0.32179991, 0.21380326, 0.70354077, 0.96757686, 0.04089591,
    0.        , 0.95487914, 0.58227088, 0.04460894, 0.99787025,
    0.00853126, 0.98697567, 0.99118411, 0.98481932, 0.99247238,
    0.09027667, 0.3654198 , 0.93497746, 0.04179224, 0.21050083
]
    
fitness_function = feed_forward

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = weights_len

init_range_low = 0.4
init_range_high = 0.6

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 70

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
    gene_space=[range(0, 1) for i in range(len(initial_weights))],
    random_seed=0,
)

ga_instance.run()

plt.plot(losses)
plt.savefig("snn_loss.png")
plt.clf()
plt.plot(accuracies)
plt.savefig("snn_acc.png")
plt.show()
