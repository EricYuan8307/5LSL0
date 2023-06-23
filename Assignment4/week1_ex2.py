import torch
import torch.nn as nn
import MNIST_dataloader
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm

def imagepre(n, x_noisy, x_output, x_clean, file_name):
    place = "images/" + file_name
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # noisy
        ax = plt.subplot(3, n, i + 1)
        ax.imshow(x_noisy[i], cmap="gray")
        ax.axis("off")

        # output image
        ax = plt.subplot(3, n, i + 1 + n)
        ax.imshow(x_output[i], cmap="gray")
        ax.axis("off")

        # original
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        ax.imshow(x_clean[i], cmap="gray")
        ax.axis("off")

    plt.savefig(place)
    plt.show()

def img_loss(train_losses, test_losses, file_name):
    num_epochs = len(train_losses)
    place = "images/" + file_name
    # plot the loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_losses, label='Training loss')
    ax.plot(test_losses, label='Testing loss')
    ax.set_xlim(0, num_epochs-1)

    # axis labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(place)
    plt.show()

class LISTA(nn.Module):
    def __init__(self,):
        super(LISTA ,self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5),stride=1, padding=2)
        self.norm = nn.BatchNorm2d(1)
        self.lambd = nn.Parameter(torch.tensor([2.0, 1.0, 0.0]))

    def _shrinkage(self, x, lumdb):
        out = x + 0.5 * (torch.sqrt((x - lumdb) * (x - lumdb) + 1) - torch.sqrt((x + lumdb) * (x + lumdb) + 1))
        return out

    def forward(self, y):
        x_k = y
        x = self.conv(x_k)
        x = self.norm(x)
        x = self._shrinkage(x, self.lambd[0])
        x = self.conv(x)
        o12 = self.norm(x)


        x = self.conv(x_k)
        x = self.norm(x)
        x = self._shrinkage(o12 + x, self.lambd[1])
        x = self.conv(x)
        o22 = self.norm(x)


        x = self.conv(x_k)
        x = self.norm(x)
        x = self._shrinkage(o22 + x, self.lambd[2])

        return x

def loss_t(model, test_data, criterion):
    device = torch.device('mps')
    model.eval()
    loss = 0

    for (x_clean, x_noisy, __) in tqdm(test_data):
        x_clean = x_clean.to(device)
        x_noisy = x_noisy.to(device)
        model = model.to(device)
        test_out = model(x_noisy)

        loss += criterion(test_out, x_clean).item()

    loss0 = loss/len(test_data)
    return loss0

def training(model, train_data, test_data, epoch, path):
    device = torch.device('mps')

    train_loss = []
    test_loss = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epo in range(epoch):
        loss_train = 0
        loss_test = 0

        for (x_clean, x_noisy, __) in tqdm(train_data):
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)
            model = model.to(device)
            x_out = model(x_noisy)
            loss = criterion(x_out, x_clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        loss_train = loss_train / len(train_data)

        loss_test = loss_t(model, test_data, criterion)
        print("train loss", loss_train)
        print("test loss",loss_test)

        train_loss.append(loss_train)
        test_loss.append(loss_test)

    torch.save(model.state_dict(), f"{path}_epochs.pth")
    img_loss(train_loss, test_loss, "ex2_2a")


def main():
    # define parameters
    data_loc = '5LSL0-Datasets'

    # get dataloaders
    batch_size = 64

    # get dataloader
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    x_clean_test = test_loader.dataset.Clean_Images
    x_noisy_test = test_loader.dataset.Noisy_Images

    # hyperparameters
    path = "5LSL0_team8"
    epoch = 20

    model = LISTA()
    train = training(model, train_loader, test_loader, epoch, path)

    # Question 2b:
    model1 = LISTA()
    model1.load_state_dict(torch.load("5LSL0_team8_epochs.pth"))
    model1.eval()
    x_output = model1(x_noisy_test)

    x_model_out = x_output.detach().numpy()
    x_model_out = x_model_out[:10, :, :].squeeze()
    x_clean_example = x_clean_test[:10, :, :, :].squeeze()  # torch.Size([10, 32, 32])
    x_noisy_example = x_noisy_test[:10, :, :, :].squeeze()  # torch.Size([10, 32, 32])

    imagepre(10, x_noisy_example, x_model_out, x_clean_example, "w1_ex2.png")

    # Question 2c:
    criterion = nn.MSELoss()
    loss = 0
    LISTA_mse_losses = 0
    for (x_clean, x_noisy, __) in test_loader:
        x_lista = model1(x_noisy)
        loss = criterion(x_lista, x_clean)
        LISTA_mse_losses += loss.item()

    print(f'test_loss = {LISTA_mse_losses / len(test_loader)}') # 0.04424767257871142


if __name__ == "__main__":
    main()
