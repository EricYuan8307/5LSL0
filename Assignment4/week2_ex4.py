import torch
import torch.nn as nn
from Fast_MRI_dataloader import create_dataloaders
import matplotlib.pyplot as plt
from torch.fft import fft2, fftshift, ifft2, ifftshift


# F
def mri_to_kspace(img, ):
    kspace = fftshift(fft2(img))
    return kspace

# F^-1
def kspace_to_mri(ksp,):
    output_mri = ifft2(ksp)
    return torch.abs(output_mri)

def kspace_to_mri_pre(ksp,):
    output_mri1 = ifftshift(ksp, dim=(1, 2))
    output_mri2 = ifft2(output_mri1, dim=(1, 2))
    return torch.abs(output_mri2)

def mul(ksp, m):
    output_mul = torch.mul(m, ksp)
    return output_mul

def softthreshold(x, gamma):

    output = torch.sign(x) * torch.max(torch.abs(x) - gamma, torch.zeros_like(x))
    return output

def mri_ista(mu, threshold, K, ksp_input, Mask):
    """
    :param mu: step size
    :param threshold:  shrinkage parameter
    :param K: number of iteration
    :param ksp_input: partial k-space measurements
    :param M: measurement mask
    :return:
    """
    mri_image = kspace_to_mri(ksp_input)
    x_t = mri_image
    FY = ksp_input

    for i in range(K):

        x_t = softthreshold(x_t, threshold)
        FX = mri_to_kspace(x_t)

        MFX = mul(FX, Mask)
        MFY = mul(FY, Mask)

        part_k_space = FX - mu*MFX + mu*MFY
        x_t = kspace_to_mri(part_k_space)

    return x_t

def imagepre(n, accel_mri, ista_mri, gt, file_name):
    place = "images/" + file_name
    plt.figure(figsize=(10, 8))
    for i in range(n):
        plt.subplot(3, 5, i + 1)
        plt.imshow(accel_mri[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Accelerated MRI')

        plt.subplot(3, 5, i + 6)
        plt.imshow(ista_mri[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('ISTA reconstruction')

        plt.subplot(3, 5, i + 11)
        plt.imshow(gt[i, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('Ground-truth MRI')

    plt.savefig(place)
    plt.show()




def main():
    # parameters
    mu = 0.8
    shrinkage = 0.15
    K = 10
    data_loc = '5LSL0-Datasets/Fast_MRI_Knee' #change the datalocation to something that works for you
    batch_size = 6
    iteration = 10

    train_loader, test_loader = create_dataloaders(data_loc, batch_size)



    # go over the dataset
    for (kspace, M, gt) in train_loader:
        continue

    ista_mri = mri_ista(mu,shrinkage,K, kspace, M)
    accel_mri = kspace_to_mri_pre(kspace)
    imagepre(5, accel_mri, ista_mri, gt, "w2_ex4b")

    # Question 4c:
    criterion = nn.MSELoss()
    loss = 0
    ISTA_MRI_mse = 0
    acc_ISTA_MRI_mse = 0

    for (kspace_t, M_t, gt_t) in test_loader:
        ista_mri = mri_ista(mu,shrinkage,K, kspace_t, M_t)
        loss = criterion(ista_mri, gt_t)
        ISTA_MRI_mse += loss.item()

    for (kspace_t, M_t, gt_t) in test_loader:
        accel_mri = kspace_to_mri_pre(kspace_t)
        loss = criterion(accel_mri, gt_t)
        acc_ISTA_MRI_mse += loss.item()

    print(f'ISTA MRI MSE = {ISTA_MRI_mse / len(test_loader)}')  # 0.02037549323243339
    print(f'Accelerated ISTA MRI MSE = {acc_ISTA_MRI_mse / len(test_loader)}')  # 0.015478021510672279


if __name__ == "__main__":
    main()