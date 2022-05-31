import torch
import torchvision
import pandas as pd
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split


# hyperparameters
batch_size = 32

# for CNN model (conv2d x2 + fc x2)
conv1c = 16
conv2c = 32
conv1k = 5
conv2k = 3
fc1 = 128
fc2 = 10
batchnorm = True
dropout = 0.1

# for training
lr = 0.001
weight_decay = 1e-5

class CNNModel(nn.Module):
    def __init__(self, conv1c, conv2c, conv1k, conv2k, fc1, fc2, batchnorm, dropout):
        super(CNNModel, self).__init__()        
        self.conv1 = nn.Conv2d(1, conv1c, kernel_size=(conv1k,conv1k))
        self.conv2 = nn.Conv2d(conv1c, conv2c, kernel_size=(conv2k,conv2k))        
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.dropout = nn.Dropout(p = dropout)     
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(conv1c)
            self.bn2 = nn.BatchNorm2d(conv2c)        
        final_size = ((28-conv1k+1)//2-conv2k+1)//2
        self.fc1 = nn.Linear(conv2c*final_size*final_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        
    def forward(self, x):
        # x: [batch_size, 1, 28, 28], assume conv1k=5 and conv2k=3
        out = self.conv1(x) # [batch_size, conv1c, 24, 24]
        out = F.relu(self.pool(out)) # [batch_size, conv1c, 12, 12]
        if self.batchnorm:
            out = self.bn1(out) # [batch_size, conv1c, 12, 12]
        out = self.conv2(out) # [batch_size, conv2c, 10, 10]
        out = F.relu(self.pool(out)) # [batch_size, conv2c, 5, 5]        
        if self.batchnorm:
            out = self.bn2(out) # [batch_size, conv2c, 5, 5]
        out = out.reshape(out.shape[0], -1) # [batch_size, conv2c*25]
        out = F.relu(self.fc1(out)) # [batch_size, fc1]
        out = self.dropout(out)
        out = F.log_softmax(self.fc2(out), dim=1) # [batch_size, fc2]
        return out
model_cnn = CNNModel(conv1c, conv2c, conv1k, conv2k, fc1, fc2, batchnorm, dropout)

@torch.no_grad()
def evaluation(model, evalloader):
    conf_mat = np.zeros((10, 10))
    model.eval()
    misclassified = []
    predicts = []
    numT = 0
    numF = 0
    for i, x in enumerate(evalloader):
        image, label = x
        pred = torch.argmax(model(image), dim=1)
        _T = torch.sum(pred == label).item()
        numT += _T
        numF += len(label) - _T
        for j in range(len(label)):
            conf_mat[label[j], pred[j]] += 1
            if label[j] != pred[j]:
                misclassified.append((image[j], label[j], pred[j]))
            predicts.append([label[j].item(),pred[j].item()])
    model.train()
    TP=0
    TN=0
    FP=0 
    FN=0
    total=np.sum(conf_mat)
    for i in range(10):
        tp=0
        tn=0
        fp=0
        fn=0
        tp += conf_mat[i][i]
        for j in range(4):
            if j!=i:
                fn += conf_mat[i][j]
                fp += conf_mat[j][i]
        tn += (total-tp-fn-fp)
        TP+=tp
        TN+=tn
        FP+=fp
        FN+=fn   
    return numT/(numT+numF), TP/(TP+FP), TP/(TP+FN), conf_mat, misclassified, predicts

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

#load check_points
model_cnn, optim, epoch, loss = load_checkpoint(model_cnn, optimizer=None, path="./checkpoints/model.pt")  

# Download MNIST dataset (or load the directly if you have already downloaded them previously)
if os.path.exists("../data/MNIST"):
    _dl = False
else:
    _dl = True
transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
trainset_all = torchvision.datasets.MNIST('../data', train=True, download=_dl, transform=transform)
testset = torchvision.datasets.MNIST('../data', train=False, download=_dl, transform=transform)
samples = []

SEED = 20211111
valid_size = 10000
train_size = len(trainset_all) - valid_size
trainset, valset = random_split(trainset_all, [train_size, valid_size], generator=torch.Generator().manual_seed(SEED))
trainloader = DataLoader(trainset, batch_size, shuffle = True, drop_last = True)
valloader = DataLoader(valset, batch_size, shuffle = False, drop_last = False)
testloader = DataLoader(testset, batch_size, shuffle = False, drop_last = False)
print('----------model----------')
print("trainset size: ", len(trainset))
print("validset size: ", len(valset))
print("testset size: ", len(testset))


#results on validation set
valacc, valpre, valrec, valcm, valmis, valpred = evaluation(model_cnn, valloader)
valcm = pd.DataFrame(valcm, dtype=int)
print("Accuracy on validset: ", valacc)
print("Precision on validset: ", valpre)
print("Recall on validset: ", valrec)
print("CNN validset Confusion Matrix: \n", valcm)

#results on testing set
tesacc, tespre, tesrec, tescm, tesmis, tespred = evaluation(model_cnn, testloader)
tescm = pd.DataFrame(tescm, dtype=int)
print("Accuracy on testset: ", tesacc)
print("Precision on testset: ", tespre)
print("Recall on testset: ", tesrec)
print("CNN Testset Confusion Matrix: \n", tescm)
name=['label','predict']
q=pd.DataFrame(columns=name, data=tespred)
q.to_csv('mnist_test_prediction.csv',index=False)

#Classifying your custom handwritten digits   
path = "./handwritten_v3"
print()
print('----------my dataset----------')
images_path = os.listdir(path)
images = []
for i in images_path:
    label = int(i.split('_')[1].split('.')[0])
    image = torchvision.io.read_image(os.path.join(path, i)).float()[0:3,:,:]
    image = image.mean(dim=0).unsqueeze(0)
    images.append((image, label))
_values = torch.concat([i for i,j in images]).reshape(-1)
_mean = _values.mean().item()
_sd = _values.std().item()
print("Mean: ",_mean, "  Std:",_sd)
mytransform = torchvision.transforms.Normalize((_mean), (_sd))
# mytransform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
for i in range(len(images)):
    img = mytransform(images[i][0])
    images[i] = (img, images[i][1])    
class mySet(Dataset):
    def __init__(self, images):
        super(mySet, self).__init__()
        self.data = images
    def __getitem__(self, x):
        return self.data[x]
    def __len__(self):
        return len(self.data)
myevalset = mySet(images)
print("My dataset size: ", len(myevalset))
myloader = DataLoader(mySet(images), shuffle=False, drop_last = False, batch_size = batch_size) 
myacc, mypre, myrec, mycm, mymis, mypred = evaluation(model_cnn, myloader)
mycm = pd.DataFrame(mycm, dtype=int)
print("Accuracy on my dataset: ", myacc)
print("Precision on my dataset: ", mypre)
print("Recall on my dataset: ", myrec)
print("CNN my dataset Confusion Matrix: \n", mycm)