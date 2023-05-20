import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchmetrics.classification import BinaryAccuracy
# import time


if torch.backends.mps.is_available():
    device = "mps"
    print(device)
elif torch.cuda.is_available():
    device = "cuda"
    print(torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print(device)


class IOT23Dataset(Dataset):

    def __init__(self, dfX, dfY):
        self.data = torch.FloatTensor(dfX.values)
        self.label = torch.FloatTensor(np.squeeze(dfY.values))
        # print(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class DNN(nn.Module):
    def __init__(self, input_dim=14):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),    
            nn.LeakyReLU(),
            nn.Linear(2, 1),    
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x
        
model = DNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.RAdam(model.parameters(), lr=0.0001)
batch_size = 256
acc = BinaryAccuracy().to(device)

num_epoch = 10

label_data = pd.read_table('adv_label.csv', sep=',', header=None, comment='#', engine='python')

for eps in np.arange(0.01, 0.11, 0.01):
    model = DNN().to(device)
    print("Loading eps={:1.2f}_adv_concat.csv...".format(eps))
    data_tst = pd.read_table('eps={:1.2f}_adv_concat.csv'.format(eps), sep=',', header=None, comment='#', engine='python')
    x, y = shuffle(data_tst, label_data, random_state=1)
    train_set = IOT23Dataset(dfX=x, dfY=y)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # print("train_set: ", len(train_set))
    # print("train_loader: ", len(train_loader))

    best_epoch = 0
    best_acc = 0.0
    print("Starting Training...")
    model.train()
    for epoch in range(num_epoch):
        # train_corrects = 0
        train_acc = 0.0
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward() 
            optimizer.step()
            train_acc += acc(outputs, labels).item()
            train_loss += batch_loss.item()
        train_acc = train_acc/len(train_loader)
        train_loss = train_loss/len(train_loader)
        print('Epoch: {}/{} Acc={:1.6f} Loss={:1.6f}'.format(epoch+1, num_epoch, train_acc, train_loss))
        if (train_acc >= best_acc):
            best_epoch = epoch + 1
            best_acc = train_acc
            torch.save(model, 'eps={:1.2f}_model.pth'.format(eps))
    print("Model saving at Epoch: {} Acc: {:1.6f}".format(best_epoch, best_acc))

    test_set = IOT23Dataset(dfX=x, dfY=y)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

    print("Starting Testing...")
    model = torch.load('eps={:1.2f}_model.pth'.format(eps))
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels) 
            test_acc += acc(outputs, labels).item()
            test_loss += batch_loss.item()
    test_acc = test_acc/len(test_loader)
    test_loss = test_loss/len(test_loader)
    print('eps={:1.2f} Acc: {:1.6f} Loss: {:1.6}'.format(eps, test_acc, test_loss))
    with open('eps_self-test.txt', 'a') as f:
        f.write('eps={:1.2f} Acc: {:1.6f} Loss: {:1.6}\n'.format(eps, test_acc, test_loss))
    


