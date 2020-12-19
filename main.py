import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import myModels
import experiment

# setting
epoch_train = 5
epoch_knockoff = 10
dataset = 'MNIST' # ['MNIST', 'FMNIST', 'KMNIST']


print("[+] Initialize Models...")
torch.manual_seed(2020) # set manual seed for reproducibility

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_codes = ['A', 'B', 'D', 'R'] # MyModelA, MyModelB, MyDeepModel, GrayResnet18

origin = { }
origin['A'] = myModels.MyModelA()
origin['B'] = myModels.MyModelB()
origin['D'] = myModels.MyDeepModel()
origin['R'] = myModels.GrayResnet18()

knockoff = { T:{ } for T in model_codes }

for T in model_codes:
    knockoff[T]['A'] = myModels.MyModelA()
    knockoff[T]['B'] = myModels.MyModelB()
    knockoff[T]['D'] = myModels.MyDeepModel()
    knockoff[T]['R'] = myModels.GrayResnet18()

if device == 'cuda':
    for T1 in model_codes:
        for T2 in model_codes:
            knockoff[T1][T2].cuda()
        origin[T1].cuda()

print("[+] Load Dataset...")
# data
train_data_MNIST = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data_MNIST = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_data_FMNIST = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data_FMNIST = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_data_KMNIST = datasets.KMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data_KMNIST = datasets.KMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# loader
train_loader_MNIST = torch.utils.data.DataLoader(train_data_MNIST, shuffle=True, batch_size=200, pin_memory=True)
test_loader_MNIST = torch.utils.data.DataLoader(test_data_MNIST, batch_size=200, pin_memory=True)

train_loader_FMNIST = torch.utils.data.DataLoader(train_data_FMNIST, shuffle=True, batch_size=200, pin_memory=True)
test_loader_FMNIST = torch.utils.data.DataLoader(test_data_FMNIST, batch_size=200, pin_memory=True)

train_loader_KMNIST = torch.utils.data.DataLoader(train_data_KMNIST, shuffle=True, batch_size=200, pin_memory=True)
test_loader_KMNIST = torch.utils.data.DataLoader(test_data_KMNIST, batch_size=200, pin_memory=True)

# choose dataset
if dataset == 'MNIST':
    train_loader = train_loader_MNIST
    test_loader = test_loader_MNIST
elif dataset == 'FMNIST':
    train_loader = train_loader_FMNIST
    test_loader = test_loader_FMNIST
elif dataset == 'KMNIST':
    train_loader = train_loader_KMNIST
    test_loader = test_loader_KMNIST

print(f"[+] {dataset} is chosen")

print("[+] Train Models...")
global_st = time.time()

train_loss, test_acc = {}, {}
for T in model_codes:
    train_loss[T], test_acc[T] = experiment.train_model(origin[T], train_loader, test_loader, num_epochs=epoch_train)

print("[+] Knockoff Models...")
knockoff_loss = { T:{ } for T in model_codes }
knockoff_acc = { T:{ } for T in model_codes }
for T1 in model_codes:
    for T2 in model_codes:
        print(f" - distillating knowledge from <model{T1}> to <model{T2}> with random noise")
        knockoff_loss[T1][T2], knockoff_acc[T1][T2] = experiment.imitate_model(origin[T1], knockoff[T1][T2], train_loader, test_loader, num_epochs=epoch_knockoff)

print(f"elapsed {time.time() - global_st}sec")

print("[+] plot graphs...")
for T1 in model_codes:
    plt.close('all')
    plt.title(f'Knockoff Model {T1}(with acc. {test_acc[T1][-1]}%) to ...')
    plt.axhline(y=test_acc[T1][-1], linestyle='-.')
    for T2 in model_codes:
        plt.plot(1 + np.arange(len(knockoff_acc[T1][T2])), knockoff_acc[T1][T2], '-o', label=f'{T1} → {T2}')
    plt.legend(loc='best')
    plt.ylabel('accuracy(%)')
    plt.xlabel('epoch(#)')
    plt.savefig(f'./img/{dataset}-{T1}.png', bbox_inches='tight')

print("[+] draw heatmap...")
softmax = nn.Softmax(dim=1)
for T in model_codes:
    x = torch.randn([100, 1, 28, 28])
    x = x.to(device)
    y = origin[T](x)
    y = softmax(y).to('cpu').detach()
    plt.close('all')
    plt.figure(figsize=(10,4))
    plt.title(f'Output Heatmap on Random Input, Model {T}')
    sns.heatmap(torch.transpose(y,0,1))
    plt.savefig(f'./img/{dataset}-{T}-heatmap.png', bbox_inches='tight')

print("[+] record datalogs...")
with open(f"./record-{dataset}.txt", "wt") as f:
    f.write("knockoff_acc : ")
    f.write(str(knockoff_acc) + "\n")
    f.write("knockoff_loss : ")
    f.write(str(knockoff_loss) + "\n")
    f.write("train_loss : ")
    f.write(str(train_loss) + "\n")
    f.write("test_acc : ")
    f.write(str(test_acc) + "\n")

