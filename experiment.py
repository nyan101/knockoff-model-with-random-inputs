import torch
import torch.nn as nn
import torch.optim as optim

import time

def train_model(model, train_loader, test_loader, num_epochs):
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss = log(softmax) + NLLloss

    train_loss = []
    test_acc = []

    st = time.time()
    print(f"training total {num_epochs} epochs...")
    for epoch in range(num_epochs):
        tot_loss = 0
        model.train()
        for idx, (x,y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
            loss = criterion(y_pred, y)
            tot_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0:
                train_loss.append(loss.item())
        print(f"Epoch {epoch+1:2d}/{num_epochs:2d} : loss {tot_loss} (elapsed {time.time() - st}s)")

        model.eval()
        acc, tot = 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
            acc += (y==y_pred.argmax(1)).sum()
            tot += len(y)
        print(f"              test acc. {acc}/{tot} ({100*acc.double()/tot:.2f}%)")
        test_acc.append(100*acc.item()/tot)
    print("done")

    return train_loss, test_acc


def imitate_model(originNet, knockoffNet, train_loader, test_loader, num_epochs):
    device = next(knockoffNet.parameters()).device
    optimizer = optim.Adam(knockoffNet.parameters())
    criterion = nn.BCELoss()
    softmax = nn.Softmax(dim=1)

    train_loss = []
    test_acc = []

    st = time.time()
    originNet.eval()
    print(f"knockoffing total {num_epochs} epochs...")
    for epoch in range(num_epochs):
        tot_loss = 0
        knockoffNet.train()
        for idx, (x,y) in enumerate(train_loader):
            x = torch.randn_like(x, device=device)
            #x = torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=device)
            y = softmax(originNet(x).detach())
            y_pred = softmax(knockoffNet(x))
            
            loss = criterion(y_pred, y)
            tot_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                train_loss.append(loss.item())
        print(f"Epoch {epoch+1:2d}/{num_epochs:2d} : loss {tot_loss} (elapsed {time.time() - st}s)")

        knockoffNet.eval()
        acc, tot = 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = knockoffNet(x)

            acc += (y==y_pred.argmax(1)).sum()
            tot += len(y)
        print(f"              test acc. {acc}/{tot} ({100*acc.double()/tot:.2f}%)")
        test_acc.append(100*acc.item()/tot)
    print("done")

    return train_loss, test_acc
