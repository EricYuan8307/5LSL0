import torch
import torch.nn as nn
import torch.optim as optim

from Fast_MRI_dataloader import create_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.fft import ifft2

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(5, 5), stride=1, padding=2)
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x

class ConvNet6e(nn.Module):
    def __init__(self):
        super(ConvNet6e, self).__init__()
        self.conv = nn.Sequential(
            ConvNet(),
            ConvNet(),
            ConvNet(),
            ConvNet(),
            ConvNet(),
        )

    def forward(self, x):
        return self.conv(x)

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

def loss_t(model, test_data, criterion):
    device = torch.device('mps')
    model.eval()
    loss = 0

    for (kspace, M, gt) in tqdm(test_data):
        acc_mri = torch.abs(ifft2(kspace))

        gt_unsqueeze = torch.unsqueeze(gt, dim=1).to(device)
        acc_mri = torch.unsqueeze(acc_mri, dim=1).to(device)
        model.to(device)

        x_out = model(acc_mri)
        loss += criterion(x_out, gt_unsqueeze).item()

    loss0 = loss/len(test_data)
    return loss0

def training(model, train_data, test_data, epoch, path):
    device = torch.device('mps')
    train_loss = []
    test_loss = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    for epo in range(epoch):
        print(f"\nTraining Epoch {epo}:")

        loss_train = 0
        loss_test = 0

        # go over all minibatches
        for i, (kspace, M, gt) in enumerate(tqdm(train_data)):

            # unsqueeze to add channel dimension, (N, 320, 320) -> (N, 1, 320, 320)
            gt_unsqueeze = torch.unsqueeze(gt, dim=1)
            kspace_unsqueeze = torch.unsqueeze(kspace, dim=1)

            acc_mri = ifft2(kspace_unsqueeze)
            acc_mri = torch.abs(acc_mri)

            # move to device
            gt_unsqueeze = gt_unsqueeze.to(device)
            acc_mri = acc_mri.to(device)
            model = model.to(device)

            # forward pass
            x_out = model(acc_mri)
            loss = criterion(x_out, gt_unsqueeze)

            # backward pass, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to the total loss
            loss_train += loss.item()

        # calculate validation loss
        loss_train /= len(train_data)
        loss_test = loss_t(model, test_data, criterion)
        print(loss_train)
        print(loss_test)

        train_loss.append(loss_train)
        test_loss.append(loss_test)
        torch.save(model.state_dict(), f"{path}_ex6e.pth")


    img_loss(train_loss, test_loss, file_name="w3_ex6e")



def main():
    data_loc = '5LSL0-Datasets/Fast_MRI_Knee'  # change the datalocation to something that works for you
    batch_size = 8
    epoch = 10
    path = "5LSL0_team8"
    model = ConvNet6e()

    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    training(model, train_loader, test_loader, epoch, path)

    # Load model:
    model.load_state_dict(torch.load("5LSL0_team8_ex6e.pth"))
    model.eval()
    criterion = torch.nn.MSELoss()

    test_los = loss_t(model, test_loader, criterion)
    print(test_los)

if __name__ == "__main__":
    main()