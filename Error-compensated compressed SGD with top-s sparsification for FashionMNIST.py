from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.random as random
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        
class Net_FC(nn.Module):
    def __init__(self):
        super(Net_FC, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


## functions for top-s sparsification
def sparsify1(atensor,sfactor): # sparsification operator for vector
    s = int(sfactor*len(atensor)) # compute s to nearest smaller integer

    if s < 1: # s is too small
        return atensor
    
    elif s == 1:
        themax = atensor.abs().max() # element of largest magnitude in atensor
        atensor[atensor.abs() < themax] = 0 # sparsify atensor
        return atensor
    
    elif s > 1:
        # flatten absolute value of atensor into a list
        alist = atensor.abs().flatten().tolist()
        alist.sort() # sort from smallest to largest
        # sparsify atensor using k-th largest element of alist
        atensor[atensor.abs() < alist[-s]] = 0
        return atensor

def sparsify2(atensor,sfactor): # sparsification operator for tensor
    size = list(atensor.size()) # get shape of atensor
    s = int(sfactor*np.prod(size)) # compute s to nearest smaller integer
    
    alist = atensor.abs().flatten().tolist()
    alist.sort()
    atensor[atensor.abs() < alist[-s]] = 0
    return atensor


def test(model, device, Xtest, ytest, b_sz):
    model.eval()
    test_loss = 0
    correct = 0
    t_sz = len(ytest)
    num_b = t_sz//b_sz
    with torch.no_grad():
        for i in range(num_b):
            data = Xtest[b_sz*i : b_sz*(i+1), ]
            target = ytest[b_sz*i : b_sz*(i+1), ]
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    test_loss /= t_sz

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, t_sz,
        100. * correct / t_sz))
    return(test_loss, 100. * correct / t_sz)


def main(b_sz,lr,sfactor): # pass in training batch size, learning rate, sfactor
    # Training settings
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(20200930)
     
    dataset_train = datasets.FashionMNIST('data', train=True, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
                        
    dataset_test = datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    
    Xtrain = dataset_train.data.float()
    Xtrain.mul_(1./255.)
    Xtrain.add_(-0.1307)
    Xtrain.mul_(0.3081)

    ytrain = dataset_train.targets
    
    Xtest = dataset_test.data.float()
    Xtest.mul_(1./255.)
    Xtest.add_(-0.1307)
    Xtest.mul_(0.3081)

    ytest = dataset_test.targets
    
    local_Xtrain = []
    local_ytrain = []
    
    for i in range(10):
    
        idx = get_indices(dataset_train, i)
        local_Xtrain.append(Xtrain[idx,])
        local_ytrain.append(ytrain[idx,])

    
    b_sz_test = 1000
    
    
    model = Net_FC().to(device)
    #model = LeNet5().to(device)

    
    num_neurons = []
    
    for idx, p in enumerate(model.parameters()):
        num_neurons.append(p.data.size())
    
    
    num_layer = len(num_neurons)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    
    iter = 1
    alpha0 = 1

    
    iter_per_epoch = 60000//(10*b_sz)
    maxepoch = 10
    #maxepoch = 30

    errloc = [] # list to store compression errors at local servers
    errcen = [] # list to store compression errors at central server
    gradloc = [] # list to store gradients at local servers
    for layer in range(num_layer):
        gradloc.append(torch.zeros(num_neurons[layer]))
        errloc += [[]] # append empty list to create nested list
        errcen += [[]]

    testloss = [] # list to store average test loss
    testacc = [] # list to store prediction accuracy

    
    for epoch in range(maxepoch):
        for k in range(iter_per_epoch):
            for i in range(10):
                st_idx = random.randint(0, 6000 - b_sz + 1)
                data = local_Xtrain[i][st_idx:st_idx+b_sz,]
                target = local_ytrain[i][st_idx:st_idx+b_sz,]
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                        
                # now the stochastic gradient is computed by the i-th dataset
                # compress the gradient (with error compensation) and send to the central server
                player = 0 # counter to keep track of layers            
                for p in model.parameters():
                    gradclone = p.grad.clone() # clone gradient

                    if k == 0:
                        if player % 2 == 0: # compress gradclone based on layer
                            gradclone = sparsify2(gradclone,sfactor)
                        elif player % 2 == 1:
                            gradclone = sparsify1(gradclone,sfactor)

                    else:
                        # add last compression error to gradient
                        gradclone.add_(errloc[player][-1])
                        if player % 2 == 0: # compress result
                            gradclone = sparsify2(gradclone,sfactor)
                        elif player % 2 == 1:
                            gradclone = sparsify1(gradclone,sfactor)

                    # aggregate local compressed gradient
                    gradloc[player].add_(gradclone) 
                    gradclone.mul_(-1)
                    p.grad.add_(gradclone) # compute compression error
                    errloc[player] += [p.grad] # store compression error
                    player += 1 # update layer counter

                
                        
            # the central server receives all compressed stochastic gradients
            # average them, then compress the averaged gradient with error compensation, and broadcast to local servers
           
            gradcen = [] # list to store gradients for central server
            for i in range(0,len(gradloc)):
                agrad = torch.Tensor(gradloc[i]) # make sure gradient is tensor
                agrad.mul_(0.1) # average each gradient
                gradcen += [agrad] # store agrad in gradcen

            player = 0
            for p in model.parameters(): # access global model
                v = gradcen[player].clone() # get gradient for this layer
                vclone = v.clone() # clone gradient
                
                if k == 0:
                    if player % 2 == 0: # compress vclone based on layer
                        v = sparsify2(v,sfactor)
                    elif player % 2 == 1:
                        v = sparsify1(v,sfactor)

                else:
                    # add last compression error to compressed gradient
                    v.add_(errcen[player][-1])
                    if player % 2 == 0:
                        v = sparsify2(v,sfactor)
                    elif player % 2 == 1:
                        v = sparsify1(v,sfactor)

                v.mul_(-1)
                vclone.add_(v) # compute central compression error
                errcen[player] = [vclone] # store compression error
                v.mul_(alpha0/(iter**0.5)) # use step size for vanilla SGD
                p.data.add_(v) # update global model
                player += 1
                
            iter += 1
            
            if k % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, k * b_sz * 10, 60000,
                    100. * k * b_sz * 10 / 60000, loss.item()))

        [aloss,anacc] = test(model, device, Xtest, ytest, b_sz_test)
        testloss += [aloss]
        testacc += [anacc]
    return(testloss,testacc)

        
if __name__ == '__main__':

    #lr = 0.01, batch size = 5, s = 0.1n, epochs = 10
    testloss, testacc = main(5,1e-2,0.1)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.1,lr=0.01,bsztest=5.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 5, s = 0.2n, epochs = 10
    testloss, testacc = main(5,1e-2,0.2)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.2,lr=0.01,bsztest=5.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 5, s = 0.5n, epochs = 10
    testloss, testacc = main(5,1e-2,0.5)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.5,lr=0.01,bsztest=5.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 5, s = 0.8n, epochs = 10
    testloss, testacc = main(5,1e-2,0.8)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.8,lr=0.01,bsztest=5.csv'
    df.to_csv(filename)
    
    #lr = 0.01, batch size = 10, s = 0.1n, epochs = 10
    testloss, testacc = main(10,1e-2,0.1)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.1,lr=0.01,bsztest=10.csv'
    df.to_csv(filename)
    
    #lr = 0.01, batch size = 10, s = 0.2n, epochs = 10
    testloss, testacc = main(10,1e-2,0.2)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.2,lr=0.01,bsztest=10.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 10, s = 0.5n, epochs = 10
    testloss, testacc = main(10,1e-2,0.5)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.5,lr=0.01,bsztest=10.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 10, s = 0.8n, epochs = 10
    testloss, testacc = main(10,1e-2,0.8)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.8,lr=0.01,bsztest=10.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 20, s = 0.1n, epochs = 10
    testloss, testacc = main(20,1e-2,0.1)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.1,lr=0.01,bsztest=20.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 20, s = 0.2n, epochs = 10
    testloss, testacc = main(20,1e-2,0.2)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.2,lr=0.01,bsztest=20.csv'
    df.to_csv(filename)
    
    #lr = 0.01, batch size = 20, s = 0.5n, epochs = 10
    testloss, testacc = main(20,1e-2,0.5)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.5,lr=0.01,bsztest=20.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 20, s = 0.8n, epochs = 10
    testloss, testacc = main(20,1e-2,0.8)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.8,lr=0.01,bsztest=20.csv'
    df.to_csv(filename)

    
    #lr = 0.01, batch size = 10, s = 0.5n, epochs = 30
    testloss, testacc = main(10,1e-2,0.5)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.5,lr=0.01,bsztest=10,epochs=30.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 10, s = 0.8n, epochs = 30
    testloss, testacc = main(10,1e-2,0.8)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.8,lr=0.01,bsztest=10,epochs=30.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 5, s = 0.5n, epochs = 30
    testloss, testacc = main(5,1e-2,0.5)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.5,lr=0.01,bsztest=5,epochs=30.csv'
    df.to_csv(filename)

    #lr = 0.01, batch size = 5, s = 0.8n, epochs = 30
    testloss, testacc = main(5,1e-2,0.8)
    results = [testloss, testacc]
    labels = ['test loss', 'prediction accuracy']
    thedict = {}
    for i in range(0,len(results)):
        thedict[labels[i]] = [str(item) for item in results[i]]
    df = pd.DataFrame(thedict)
    filename = 'b=0.8,lr=0.01,bsztest=5,epochs=30.csv'
    df.to_csv(filename)
