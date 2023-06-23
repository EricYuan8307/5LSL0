import torch
import numpy as np
import MNIST_dataloader
import matplotlib.pyplot as plt

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


def softthreshold(x, gamma): # do we need to use the absolute?
    H, W = x.shape

    for i in range(H):
        for j in range(W):
            if x[i,j] > gamma:
                x[i,j] = (x[i,j] - gamma)
            elif gamma >= x[i,j] >= -gamma:
                x[i,j] = 0
            else:
                x[i,j] = (x[i,j] + gamma)

    return x

def ista(mu, shrinkage, K, input):
    H, W = input.shape[1:]
    A = np.identity(H)
    I = np.identity(H)
    x_k = np.zeros((H,W))
    image_list = []


    for y in input:
        for j in range(K):
            z = np.dot((I - mu * np.dot(A.T, A)), x_k) + mu * np.dot(A, y)

            # soft thresholding
            x_k = softthreshold(z, shrinkage)

        # store the results
        image_list.append(x_k)

    x_out = torch.from_numpy(np.array(image_list)).float()
    return x_out

def loss_ista(data, mu, shrinkage, K):
    #initial loss
    loss = 0
    mse = torch.nn.MSELoss()

    for (x_clean, x_noisy, __) in data:
        x_out = ista(mu, shrinkage, K, x_noisy.squeeze())
        loss += mse(x_out.squeeze(), x_clean.squeeze()).item()

    return loss/len(data)


def main():

    data_loc = '5LSL0-Datasets' #change the datalocation to something that works for you
    batch_size = 64

    # get dataloader
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    # get some examples
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)

    x_clean = x_clean_example[:10].squeeze()
    x_noisy = x_noisy_example[:10].squeeze()

    # hyperparameters
    mu = 0.3
    threshold = 0.2
    K = 20
    n = 10

    # result
    x_output = ista(mu, threshold, K, x_noisy)
    imagepre(n, x_noisy, x_output, x_clean, "w1_ex1.png")
    loss = loss_ista(test_loader, mu, threshold, K)

    print(loss) # 0.45600699922841065


if __name__ == "__main__":
    main()