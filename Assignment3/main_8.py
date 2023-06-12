# %% imports
# libraries
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# local imports
import MNIST_dataloader
import variational_autoencoder

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = 'D://5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
no_epochs = 50
learning_rate = 1e-4
beta = 0.01
retrain = False
latent_size = 16 #a larger latent size works better for this exercise

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
VAE = variational_autoencoder.VAE(latent_size)

# create the optimizer
optimizer = torch.optim.Adam(VAE.parameters(),lr = learning_rate)

# %% training loop
if retrain == True:
    loss_list = []
    # go over all epochs
    for epoch in range(no_epochs):
        print(f"\nTraining Epoch {epoch}:")
        # go over all minibatches
        for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
            # reset the gradients of the optimizer
            optimizer.zero_grad()
            
            # i/o of the network
            r,mu,log_var = VAE(x_clean)
            
            # calculate the loss
            mse_loss = torch.nn.functional.mse_loss(r,x_clean)
            reconstruction_loss = mse_loss
            kl_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var),dim=1),dim=0)
            
            loss = reconstruction_loss+beta*kl_loss
            
            # perform backpropagation
            loss.backward()
            
            # do an update step
            optimizer.step()
            
            # append loss to list
            loss_list.append(loss.item())
    
    # save parameters
    torch.save(VAE.state_dict(),"VAE_dict16.tar")
else:
    state_dict = torch.load("VAE_dict16.tar")
    VAE.load_state_dict(state_dict)

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels

# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]



# %% exercise 8a answer
with torch.no_grad():
    r_noisy_example,mu,log_var = VAE(x_noisy_example)

# show the examples in a plot
plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+11)
    plt.imshow(r_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("figures/exercise_8a.png",dpi=300,bbox_inches='tight')
plt.close()

# %% exercise 8b answer
# parameters
no_iterations = 1000
learning_rate = 1e-2
beta = 0.01

estimated_latent = nn.Parameter(torch.randn(10,latent_size))
optimizer_map = torch.optim.Adam([estimated_latent],lr = learning_rate)

# optimization
loss_list = []
for i in tqdm(range(no_iterations)):
    optimizer_map.zero_grad()
    
    reconstruction = VAE.decoder(estimated_latent)
    loss = torch.mean((x_noisy_example-reconstruction)**2) + beta*torch.mean(estimated_latent**2)
    
    loss.backward()
    optimizer_map.step()
    
    # append loss to list
    loss_list.append(loss.item())

# loss plot
plt.figure()
plt.plot(loss_list)
plt.grid()
plt.xlabel('iterations')
plt.ylabel('MAP loss')
plt.savefig("figures/exercise_8b_loss.png",dpi=300,bbox_inches='tight')
plt.close()

# show the estimated images
plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+11)
    plt.imshow(reconstruction[i,0,:,:].detach(),cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("figures/exercise_8b_examples.png",dpi=300,bbox_inches='tight')
plt.close()