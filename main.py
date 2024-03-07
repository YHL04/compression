

import torch
import torch.nn.functional as F
import torchvision

from model import Model


def main():
    num_epochs = 10
    root = ''
    B = 32
    lr = 1e-4
    device = 'cuda:0'

    model = Model()

    traindataset = torchvision.datasets.ImageNet(root=root, split='train')
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=B, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs, c = model(inputs)
            loss = F.mse(outputs, labels) + torch.pow(c, 2)

            loss.backward()
            optimizer.step()


    torch.save(model.state_dict(), "saved/final")
