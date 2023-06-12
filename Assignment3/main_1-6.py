# %% imports
# libraries
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# local imports
import MNIST_dataloader
import autoencoder

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = 'D://5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
no_epochs = 10
learning_rate = 1e-4

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
AE = autoencoder.AE()

# create the optimizer
optimizer = torch.optim.Adam(AE.parameters(),lr = learning_rate)

# %% training loop
loss_list = []

# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # reset the gradients of the optimizer
        optimizer.zero_grad()
        
        # i/o of the network
        r,h = AE(x_clean)
        
        # calculate the loss
        loss = torch.nn.functional.mse_loss(r, x_clean)
        
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
    plt.ylabel('mse')
    plt.grid()
    plt.title('training loss')
    plt.tight_layout()
    plt.savefig("figures/exercise_1_loss.png",dpi=300,bbox_inches='tight')
    plt.close()

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
    r_clean_example,_ = AE(x_clean_example)

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
plt.savefig("figures/exercise_1_examples.png",dpi=300,bbox_inches='tight')
plt.close()


# %% Exercise 2 answer
with torch.no_grad():
    _,h = AE(x_clean_test)

plt.figure()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i in range(10):
    plt.scatter(h[labels_test==i,0,0,0],h[labels_test==i,0,1,0],c=colors[i],s=7)

plt.legend(torch.arange(10).numpy(),loc=0)
plt.grid()
plt.tight_layout()
plt.savefig("figures/exercise_2.png",dpi=300,bbox_inches='tight')
plt.close()

# %% Exercise 3 answer
# first encode both sets
with torch.no_grad():
    _,h_train = AE(x_clean_train)
    _,h_test = AE(x_clean_test)
    
h_train = h_train.squeeze()
h_test = h_test.squeeze()
    
# do the nearest neighbour search
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(h_train)
distances, indices = nbrs.kneighbors(h_test)

nnClassification = labels_train[indices[:]]

# figure out what the accuracies are
nnCorrect = torch.zeros(10)
nnTotal = torch.zeros(10)
nnAccuracies = torch.zeros(10)

print("\nnn Classification accuracies are:")
for i in range(10):
    nnCorrect[i] = torch.sum(nnClassification[labels_test == i,0] == i)
    nnTotal[i] = torch.sum(labels_test == i)
    nnAccuracies[i] = 100*nnCorrect[i]/nnTotal[i]
    print(f"for digit {i}, accuracy is {nnAccuracies[i]:.3}")
    
# %% Exercise 4 answer
Classifier = autoencoder.Classifier()
optimizer_classifier = torch.optim.Adam(Classifier.parameters(),lr = learning_rate)

loss_list = []
test_loss_list = []

# get the test loss
with torch.no_grad():
    y = Classifier(x_clean_test)
    loss = torch.nn.functional.cross_entropy(y, labels_test)  
    test_loss_list.append(loss.item())

# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # reset the gradients of the optimizer
        optimizer_classifier.zero_grad()
        
        # i/o of the network
        y = Classifier(x_clean)
        
        # calculate the loss
        loss = torch.nn.functional.cross_entropy(y, label)
        
        # perform backpropagation
        loss.backward()
        
        # do an update step
        optimizer_classifier.step()
        
        # append loss to list
        loss_list.append(loss.item())
        
    # get the test loss
    with torch.no_grad():
        y = Classifier(x_clean_test)
        loss = torch.nn.functional.cross_entropy(y, labels_test)  
        test_loss_list.append(loss.item())
    
    # plot loss over time
    e = torch.arange(len(loss_list))/len(train_loader)
    
    plt.figure()
    plt.plot(e,loss_list)
    plt.plot(test_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('CE')
    plt.grid()
    plt.title('exercise 4 losses')
    plt.legend(('train','test'))
    plt.tight_layout()
    plt.savefig("figures/exercise_4.png",dpi=300,bbox_inches='tight')
    plt.close()
    
# get the accuracies on the test set
predictions = torch.argmax(y,dim=-1)

ex4Correct = torch.zeros(10)
ex4Total = torch.zeros(10)
ex4Accuracies = torch.zeros(10)

print("\nExercise 4 classification accuracies are:")
for i in range(10):
    ex4Correct[i] = torch.sum(predictions[labels_test == i] == i)
    ex4Total[i] = torch.sum(labels_test == i)
    ex4Accuracies[i] = 100*ex4Correct[i]/ex4Total[i]
    print(f"for digit {i}, accuracy is {ex4Accuracies[i]:.3}")
    
# %% exercise 5 answer
image_grid = torch.zeros(32*15,32*15)

for x in range(15):
    for y in range(15):
        h = torch.zeros(1,1,2,1)
        h[0,0,0,0] = x/15*20+5
        h[0,0,1,0] = y/15*20
        
        with torch.no_grad():
            r = AE.decoder(h)
            image_grid[(14-y)*32:(15-y)*32,x*32:(x+1)*32] = r[0,0,:,:]
            
plt.figure()
plt.imshow(image_grid,cmap='gray')
plt.xticks(np.arange(6)*32*3,np.arange(6)*3*20//15+5)
plt.yticks(np.arange(6)*32*3,-np.arange(6)*3*20//15+20)

plt.tight_layout()
plt.savefig("figures/exercise_5.png",dpi=300,bbox_inches='tight')
plt.close()

# %% exercise 6 answer
with torch.no_grad():
    r_noisy_example,_ = AE(x_noisy_example)

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
plt.savefig("figures/exercise_6.png",dpi=300,bbox_inches='tight')
plt.close()