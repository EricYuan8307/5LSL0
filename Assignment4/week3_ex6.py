from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift, ifft2, ifftshift
import torch
import torch.nn as nn
import torch.optim as optim
from Fast_MRI_dataloader import create_dataloaders

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
    model.eval()
    loss = 0

    for (kspace, M, gt) in tqdm(test_data):
        acc_mri = torch.abs(ifft2(kspace))

        x_out = model(acc_mri, M)
        loss += criterion(x_out, gt).item()

    loss0 = loss/len(test_data)
    return loss0

# F
def mri_to_kspace(img, ):
    return fft2(img)

def mri_to_kspace_acc(img):
    return fftshift(fft2(img))

# F^-1
def kspace_to_mri(ksp,):
    output_mri = ifft2(ksp)
    return torch.abs(output_mri)

def acc_mri_to_kspace(mri,):
    output_ksp = ifftshift(mri, dim=(1, 2))
    output_ksp = ifft2(output_ksp, dim=(1, 2))
    return torch.abs(output_ksp)

def mul(ksp, m):
    output_mul = torch.mul(m, ksp)
    return output_mul

def imagepre(accel_mri, ista_mri, gt, file_name):
    place = "images/" + file_name
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(accel_mri[i + 1, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Accelerated MRI')

        plt.subplot(3, 5, i + 6)
        plt.imshow(ista_mri[i + 1, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Reconstruction from ProxNet')

        plt.subplot(3, 5, i + 11)
        plt.imshow(gt[i + 1, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Ground truth')

    plt.savefig(place)
    plt.show()

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


class ProxISTA(nn.Module):
    def __init__(self, mu_init = 0.5):
        super(ProxISTA, self).__init__()
        self.network1 = ConvNet()
        self.network2 = ConvNet()
        self.network3 = ConvNet()
        self.network4 = ConvNet()
        self.network5 = ConvNet()

        self.mu = torch.full((1,), mu_init)
        self.sigmoid = nn.Sigmoid()


    def forward(self, pksp, Mask):
        x = kspace_to_mri(pksp)

        # initialize
        FY = pksp


        # layer 1
        x = torch.unsqueeze(x, dim=1)
        x = self.network1(x)
        x = torch.squeeze(x, dim=1)

        FX = mri_to_kspace_acc(x)
        MFX = mul(FX, Mask)
        MFY = mul(FY, Mask)  # torch.Size([100, 320, 320])
        mu = self.sigmoid(self.mu).item()

        part_k_space = FX - mu * MFX + mu * MFY
        x = kspace_to_mri(part_k_space)


        # layer 2
        x = torch.unsqueeze(x, dim=1)
        x = self.network2(x)
        x = torch.squeeze(x, dim=1)

        FX = mri_to_kspace_acc(x)
        MFX = mul(FX, Mask)
        MFY = mul(FY, Mask)  # torch.Size([100, 320, 320])
        mu = self.sigmoid(self.mu).item()

        part_k_space = FX - mu * MFX + mu * MFY
        x = kspace_to_mri(part_k_space)

        # layer 3
        x = torch.unsqueeze(x, dim=1)
        x = self.network3(x)
        x = torch.squeeze(x, dim=1)

        FX = mri_to_kspace_acc(x)
        MFX = mul(FX, Mask)
        MFY = mul(FY, Mask)  # torch.Size([100, 320, 320])
        mu = self.sigmoid(self.mu).item()

        part_k_space = FX - mu * MFX + mu * MFY
        x = kspace_to_mri(part_k_space)

        # # layer 4
        x = torch.unsqueeze(x, dim=1)
        x = self.network4(x)
        x = torch.squeeze(x, dim=1)

        FX = mri_to_kspace_acc(x)
        MFX = mul(FX, Mask)
        MFY = mul(FY, Mask)  # torch.Size([100, 320, 320])
        mu = self.sigmoid(self.mu).item()

        part_k_space = FX - mu * MFX + mu * MFY
        x = kspace_to_mri(part_k_space)

        #layer 5
        x = torch.unsqueeze(x, dim=1)
        x = self.network5(x)
        x = torch.squeeze(x, dim=1)

        FX = mri_to_kspace_acc(x)
        MFX = mul(FX, Mask)
        MFY = mul(FY, Mask)  # torch.Size([100, 320, 320])
        mu = self.sigmoid(self.mu).item()

        part_k_space = FX - mu * MFX + mu * MFY
        x = kspace_to_mri(part_k_space)

        return x


def training(model, train_data, test_data, epoch, path):
    train_loss = []
    test_loss = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    for epo in range(epoch):
        print(f"\nTraining Epoch {epo}:")

        loss_train = 0
        loss_test = 0
        for i, (partial_kspace, M, gt) in enumerate(tqdm(train_data)):
            out = model(partial_kspace, M)
            loss = criterion(out, gt)

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
        torch.save(model.state_dict(), f"{path}_ex6.pth")
    img_loss(train_loss, test_loss, file_name="w3_ex6")






def main():
    epoch = 10
    data_loc = '5LSL0-Datasets/Fast_MRI_Knee'  # change the datalocation to something that works for you
    batch_size = 8
    path = "5LSL0_team8"

    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    model = ProxISTA()


    training(model, train_loader, test_loader, epoch, path)

    # Load model:
    model.load_state_dict(torch.load("5LSL0_team8_ex6.pth"))
    model.eval()

    for i, (pksp, M, gt) in enumerate(test_loader):
        if i == 1:
            break

    acc_mri = kspace_to_mri(pksp)
    test_out = model(pksp,M)
    test_out = test_out.detach().numpy()

    criterion = torch.nn.MSELoss()
    test_los = loss_t(model,test_loader, criterion)

    imagepre(acc_mri, test_out, gt, file_name="w3_ex6c")
    print(test_los) # 0.3014943503564404



if __name__ == "__main__":
    main()