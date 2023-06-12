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
latent_size = 2

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
            
        # plot loss over time
        e = torch.arange(len(loss_list))/len(train_loader)
        
        plt.figure()
        plt.plot(e,loss_list)
        plt.xlabel('epoch')
        plt.ylabel('mse + kl')
        plt.grid()
        plt.title('training loss')
        plt.tight_layout()
        plt.savefig("figures/exercise_7a_loss.png",dpi=300,bbox_inches='tight')
        plt.close()
    
    # save parameters
    torch.save(VAE.state_dict(),"VAE_dict2.tar")
else:
    state_dict = torch.load("VAE_dict2.tar")
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


# %% plot some examples
with torch.no_grad():
    r_clean_example,_,_ = VAE(x_clean_example)

# show the examples in a plot
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2,10,i+11)
    plt.imshow(r_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("figures/exercise_7a_examples.png",dpi=300,bbox_inches='tight')
plt.close()

# %% exercise 7b
with torch.no_grad():
    _,mu,log_s = VAE(x_clean_test)

plt.figure()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i in range(10):
    plt.scatter(mu[labels_test==i,0],mu[labels_test==i,1],c=colors[i],s=7)

plt.legend(torch.arange(10).numpy(),loc=0)
plt.grid()
plt.tight_layout()
plt.savefig("figures/exercise_7b.png",dpi=300,bbox_inches='tight')
plt.close()

# %% exercise 7D
image_grid = torch.zeros(32*15,32*15)

for x in range(15):
    for y in range(15):
        h = torch.zeros(1,2)
        h[0,0] = x/15*6-3
        h[0,1] = y/15*6-3
        
        with torch.no_grad():
            r = VAE.decoder(h)
            image_grid[(14-y)*32:(15-y)*32,x*32:(x+1)*32] = r[0,0,:,:]
            
plt.figure()
plt.imshow(image_grid,cmap='gray')
plt.xticks(np.arange(7)*32*2.5,np.arange(7)*2.5/15*6-3)
plt.yticks(np.arange(7)*32*2.5,-np.arange(7)*2.5/15*6+3)

plt.tight_layout()
plt.savefig("figures/exercise_7d.png",dpi=300,bbox_inches='tight')
plt.close()