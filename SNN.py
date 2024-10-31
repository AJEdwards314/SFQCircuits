#!pip install snntorch

import snntorch as snn
from snntorch import spikeplot as splt
import torch

# plotting
import matplotlib.pyplot as plt
from IPython.display import HTML

#@title Plotting Settings
def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, 
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange")
  ax[0].set_ylim([0, ylim_max1])
  ax[0].set_xlim([0, 200])
  ax[0].set_ylabel("Input Current (Iin)")
  if title:
    ax[0].set_title(title)

  # Plot membrane potential
  ax[1].plot(mem)
  ax[1].set_ylim([0, ylim_max2]) 
  ax[1].set_ylabel("Membrane Potential (Umem)")
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
  ax[2].set_ylim([0, 10])

  plt.show()

def dvs_animator(spike_data):
  fig, ax = plt.subplots()
  anim = splt.animator((spike_data[:,0] + spike_data[:,1]), fig, ax)
  return anim

def find_nearest(biasCurrent, probabilities, val):
    ind = int(len(probabilities)/2)
    last = 0
    while True:
        if probabilities[ind] == val:
            return biasCurrent[ind]
        if val > probabilities[ind]:
            ind, last = int(ind+abs(ind - last)/2), ind
        else:
            ind, last = int(ind-abs(ind - last)/2), ind
        if abs(ind-last) == 1:
            return biasCurrent[ind]
        if ind < 0 or ind > len(probabilities):
            return biasCurrent[last]

# import
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
     
# dataloader arguments
batch_size = 128
data_path=r"C:\Users\ecu210001\Documents\NSC\Fields\Super Conducting\SFQ\data\mnist"

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        # time-loop
        for step in range(num_steps):
          cur1 = self.fc1(x.flatten(1)) # batch128 x 784
          spk1, mem1 = self.lif1(cur1, mem1)
          cur2 = self.fc2(spk1)
          spk2, mem2 = self.lif2(cur2, mem2)
          
          # store in list
          spk2_rec.append(spk2)
          mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0) # time-steps x batch x num_out
        
# Load the network onto CUDA if available
net = Net().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 1 # 60000 / 128 = 468
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, _ = net(data)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        loss_val = loss(spk_rec.sum(0), targets) # batch x num_out

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Print train/test loss/accuracy
        if counter % 10 == 0:
            print(f"Iteration: {counter} \t Train Loss: {loss_val.item()}")
        counter += 1

        if counter == 100:
          break

def measure_accuracy(model, dataloader):
  with torch.no_grad():
    model.eval()
    running_length = 0
    running_accuracy = 0

    for data, targets in iter(dataloader):
      data = data.to(device)
      targets = targets.to(device)

      # forward-pass
      spk_rec, _ = model(data)
      spike_count = spk_rec.sum(0) # batch x num_outputs
      _, max_spike = spike_count.max(1)

      # correct classes for one batch
      num_correct = (max_spike == targets).sum()

      # total accuracy
      running_length += len(targets)
      running_accuracy += num_correct
    
    accuracy = (running_accuracy / running_length)

    return accuracy.item()

print(f"Test set accuracy: {measure_accuracy(net, test_loader)}")

state_dict = net.state_dict()

op = open('outputProbabilities.csv')
data = op.readlines()
op.close()
biasCurrents = []
probabilities = []
for i in data[1:]:
    placeHold = i.split(',')
    biasCurrents.append(float(placeHold[0]))
    probabilities.append(float(placeHold[1]))

biasCurrents = [x for _, x in sorted(zip(probabilities, biasCurrents))]
probabilities.sort()

relevent=['fc1',"fc2"]

synNum=786

circuit = open('testOutputs.cir',"a")
counter = 1
offset = 0

for i in relevent:
    weights = state_dict[i+'.weight']
    biases = state_dict[i+'.bias']
    for j in range(len(weights)):
        for k in range(len(weights[j])):
            biasCurrent = find_nearest(biasCurrents,probabilities,abs(weights[j][k]))
            if weights[j][k] > 0:
                circuit.write("X{} synapse {} {} 0\n".format(counter, offset+j, synNum))
            else:
                circuit.write("X{} synapse {} 0 {}\n".format(counter, offset+j, synNum))
            synNum+=1
            counter+=1
        print(i,"weights written",j)
        #treat 785 as constant power
        if weights[j][k] > 0:
            circuit.write("X{} synapse 785 {} 0\n".format(counter, synNum))
        else:
            circuit.write("X{} synapse 785 0 {}\n".format(counter, synNum))
        synNum+=1
        counter+=1
        addstr = "X{} Adder ".format(counter)
        for k in range(synNum-len(weights[j])-1, synNum+1):
            addstr+='{} '.format(k)
        synNum+=1
        circuit.write(addstr+'\n')
    offset = synNum
    print(i,"done")
