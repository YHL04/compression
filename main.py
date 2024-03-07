

import torch.nn.functional as F

from dataloader import *
from model import Model


def main():
    num_epochs = 1000
    lr = 1e-4
    device = 'cuda:0'
    lambda_ = 0.01

    dim_x, dim_y, channels = 32, 32, 3
    trainloader, testloader = get_CIFAR100_dataloaders(batch_size=256, test_batch_size=256)

    model = Model(dim_x=dim_x, dim_y=dim_y, channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.load_state_dict(torch.load("saved/final"))

    i = 0
    for epoch in range(num_epochs):
        for inputs, _ in trainloader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs, c = model(inputs)

            r_loss = torch.pow((outputs - inputs) * 100_000, 2).mean()
            c_loss = torch.pow(c, 2).mean()
            # loss = r_loss + lambda_ * c_loss
            loss = r_loss

            if i % 100 == 0:
                print(r_loss.item(), c_loss.item())
            i += 1

            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), "saved/final")


if __name__ == "__main__":
    main()

