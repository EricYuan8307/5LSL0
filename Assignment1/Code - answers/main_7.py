# %% imports
# libraries
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# local files
import MNIST_dataloader

# %% data
# define parameters
data_loc = 'Datasets' #change the datalocation to something that works for you
batch_size = 64

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

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
# use these example images througout the assignment as the first 10 correspond to the digits 0-9

# %% 7a
class custom_ReLU(nn.Module):
    def __init__(self):
        super(custom_ReLU, self).__init__()
    
        
    def forward(self, x):
        x = (x>=0)*x + (x<0)*0
        return x
    
class Non_Linear_Network(nn.Module):
    def __init__(self):
        super(Non_Linear_Network, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(32**2, 16**2), # reduce the image in size somewhat
            custom_ReLU(),           # activation function
            nn.Linear(16**2, 16**2), # bottleneck 
            custom_ReLU(),           # activation function
            nn.Linear(16**2, 32**2), # back to full image size
            )
    
        
    def forward(self, x):
        # get the batch size
        batch_size = x.size(0)
        
        # vectorize
        x = x.reshape(batch_size,32**2)
        
        # use the layers
        x = self.layers(x)
        
        # reshape back to image
        x = x.reshape(batch_size,1,32,32)
        
        return x
    
# test this
custom_ReLU_instance = custom_ReLU()
ReLU_in  = torch.arange(10)-5
ReLU_out = custom_ReLU_instance(ReLU_in)

plt.figure(figsize=(4,3))
plt.plot(ReLU_in,ReLU_out)
plt.grid()
plt.xlabel('input')
plt.ylabel('output')
plt.tight_layout()
plt.savefig("figures/exercise_7a.png",dpi=300)
plt.show()

# %% 7b
non_linear_network = Non_Linear_Network()
optimizer = torch.optim.Adam(non_linear_network.parameters(),lr = 1e-3)

# %% 7c
# using no gradient calculation makes this a lot faster
with torch.no_grad():
    x_example_out = non_linear_network(x_noisy_example)

# show the examples in a plot
plt.figure(figsize=(12,4.5))
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+11)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+21)
    plt.imshow(x_example_out[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("figures/exercise_7c.png",dpi=300)
plt.show()

# %% 7d
# parameters
no_epochs = 10

# initialize two lists to save the losses during training
loss_list = []
test_loss_list  = []

# get the test loss before even doing any training
with torch.no_grad():
    x_estimate = non_linear_network(x_noisy_test)
    mse_loss = torch.nn.functional.mse_loss(x_estimate,x_noisy_test)
    test_loss_list.append(mse_loss.item())

# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    # go over all minibatches in the train set
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # reset the gradients of the optimizer
        optimizer.zero_grad()
        
        # i/o of the network
        x_estimate = non_linear_network(x_noisy)
        
        # calculate the loss
        mse_loss = torch.nn.functional.mse_loss(x_estimate,x_clean)
        
        # perform backpropagation
        mse_loss.backward()
        
        # do an update step
        optimizer.step()
        
        # append loss to list
        loss_list.append(mse_loss.item())
        
    # get the test loss
    with torch.no_grad():
        x_estimate = non_linear_network(x_noisy_test)
        mse_loss = torch.nn.functional.mse_loss(x_estimate,x_clean_test)
        test_loss_list.append(mse_loss.item())    
    
    # plot loss over time
    e = torch.arange(len(loss_list))/len(train_loader)
    plt.figure(figsize=(4,3))
    plt.plot(e,loss_list)
    plt.plot(test_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.grid()
    plt.legend(('train','test'))
    plt.tight_layout()
    plt.savefig("figures/exercise_7d.png",dpi=300,bbox_inches='tight')
    plt.close()
    
    # save the model weights
    torch.save(non_linear_network.state_dict(),"state_dict_non_linear.tar")
    
# %% 7e
# using no gradient calculation makes this a lot faster
with torch.no_grad():
    x_example_out = non_linear_network(x_noisy_example)

# show the examples in a plot
plt.figure(figsize=(12,4.5))
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+11)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+21)
    plt.imshow(x_example_out[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("figures/exercise_7e.png",dpi=300)
plt.show()