

import torch
from model import Model
from dataloader import get_CIFAR100_dataloaders
import matplotlib.pyplot as plt


def main():
    num_epochs = 100
    device = 'cuda:0'
    dim_x, dim_y, channels = 32, 32, 3

    model = Model(dim_x=dim_x, dim_y=dim_y, channels=channels)
    model.load_state_dict(torch.load("saved/final"))
    model = model.to(device)
    trainloader, testloader = get_CIFAR100_dataloaders(batch_size=1, test_batch_size=1)

    for epoch in range(num_epochs):
        for inputs, _ in trainloader:
            plt.subplot(2, 1, 1)
            plt.imshow(inputs[0].permute(1, 2, 0).clone().detach().cpu().numpy())
            plt.subplot(2, 1, 2)
            rec, c = model(inputs.to(device))
            print(c.flatten())
            plt.imshow(rec[0].permute(1, 2, 0).clone().detach().cpu().numpy())
            plt.show()


if __name__ == "__main__":
    main()

