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
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))

data_tst = pd.read_table('normalize_combine_CT.csv', sep=',', header=None, comment='#', engine='python')
label_data = pd.read_table('2label_combine.csv', sep=',', header=None, comment='#', engine='python')

# X_train, X_test, Y_train, Y_test = train_test_split(data_tst, label_data, test_size=0.1, random_state=1)
x, y = shuffle(data_tst, label_data, random_state=1)


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
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 4),    
            nn.ReLU(),
            nn.Linear(4, 1),    
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x
        
model = DNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.RAdam(model.parameters(), lr=0.00001)
batch_size = 256
acc = BinaryAccuracy().to(device)

num_epoch = 10

# train_set = IOT23Dataset(dfX=X_train, dfY=Y_train)
train_set = IOT23Dataset(dfX=x, dfY=y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
print("train_set: ", len(train_set))
print("train_loader: ", len(train_loader))
# val_set = IOT23Dataset(dfX=X_test, dfY=Y_test)
# val_loader = DataLoader(dataset=val_set, batch_size=batch_size)
# print("validate_set: ", len(val_set))
# print("validate_loader: ", len(val_loader))
# test_set = IOT23Dataset(dfX=x, dfY=y)
# test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
# print("test_set: ", len(test_set))
# print("test_loader: ", len(test_loader))

best_epoch = 0
best_acc = 0.0
print("Starting Training...")
for epoch in range(num_epoch):
    model.train()
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
        # v, train_pred = torch.max(outputs, 0)
        # train_corrects += (train_pred.cpu() == labels.cpu()).sum().item()
        # train_acc = float(train_corrects/len(train_set))
        train_acc += acc(outputs, labels).item()
        train_loss += batch_loss.item()
        # if i % 1000 == 999:  # 每 100 個 batch 輸出一次
        #     print('Epoch: {}/{} [{:03d}/{:03d}] Acc: {:1.6f} loss: {:1.6f} Best: {:1.6f}'.format(epoch+1, num_epoch, i+1, len(train_loader), train_acc, batch_loss.item(), best_acc))
    train_acc = train_acc/len(train_loader)
    train_loss = train_loss/len(train_loader)
    print('Epoch: {}/{} Acc={:1.6f} Loss={:1.6f}'.format(epoch+1, num_epoch, train_acc, train_loss))
    # model.eval()
    # # val_corrects = 0
    # val_acc = 0.0
    # val_loss = 0.0
    # with torch.no_grad():
    #     for i, data in enumerate(val_loader):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         batch_loss = criterion(outputs, labels) 
    #         # _, val_pred = torch.max(outputs, 0)
    #         # val_corrects += (val_pred.cpu() == labels.cpu()).sum().item()
    #         # val_acc = float(val_corrects/len(val_set))
    #         val_acc += acc(outputs, labels).item()
    #         val_loss += batch_loss.item()
    # val_acc = val_acc/len(val_loader)
    # val_loss = val_loss/len(val_loader)
    # print('Valid: Acc={:1.6f} Loss={:1.6f}'.format(val_acc, val_loss))
    if (train_acc >= best_acc):
        best_epoch = epoch + 1
        best_acc = train_acc
        print("Model saving at Epoch: {} Acc: {:1.6f}".format(best_epoch, best_acc))
        torch.save(model, 'model_CT.pth')
    else:
        print("Last best model at Epoch: {} Acc: {:1.6f}".format(best_epoch, best_acc))
    with open('CT_train.txt', 'a') as f:
        f.write('Epoch: {}/{} Acc={:1.6f} Loss={:1.6f} Best Epoch={} Best Acc={:1.6f}\n'.format(epoch+1, num_epoch, train_acc, train_loss, best_epoch, best_acc))


# model = torch.load('model.pth')            
# model.eval()
# print("Starting Testing...")
# # test_corrects = 0
# eps_list = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]
# for eps in eps_list:
#     test_acc = 0.0
#     test_loss = 0.0
#     for i, data in enumerate(test_loader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         inputs.requires_grad = True
#         outputs = model(inputs)
#         batch_loss = criterion(outputs, labels) 
#         model.zero_grad()
#         batch_loss.backward() 
#         data_grad = inputs.grad.data
#         adv = inputs + eps * data_grad.sign()
#         # 如要儲存對抗樣本 將下面這行打開
#         # pd.DataFrame(adv.cpu().detach().numpy()).to_csv("eps={}_adv.csv".format(eps), header=False, index=False, mode='a')
#         outputs = model(adv)
#         # _, test_pred = torch.max(outputs, 0)
#         # test_corrects += (test_pred.cpu() == labels.cpu()).sum().item()
#         # test_acc = float(test_corrects/len(test_set))
#         test_acc += acc(outputs, labels).item()
#         test_loss += batch_loss.item()
#         # if i % 10 == 9:  # 每 10 個 batch 輸出一次
#         #     print('[{:03d}/{:03d}] Acc: {:1.6f} loss: {:1.6f}'.format(i+1, len(test_loader), acc(outputs, labels).item(), batch_loss.item()))
#     print('eps={} Acc: {:1.6f} loss: {:1.6f}'.format(eps, test_acc/len(test_loader), test_loss/len(test_loader)))
    


