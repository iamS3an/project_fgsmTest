import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchmetrics.classification import BinaryAccuracy


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

data_tst = pd.read_table('normalize_combine.csv', sep=',', header=None, comment='#', engine='python')
label_data = pd.read_table('2label_combine.csv', sep=',', header=None, comment='#', engine='python')

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
test_set = IOT23Dataset(dfX=x, dfY=y)
pd.Series(0, dtype=int, index=np.arange(len(test_set))).to_csv("adv_label.csv", header=False, index=False, mode='w')
pd.Series(1, dtype=int, index=np.arange(len(test_set))).to_csv("adv_label.csv", header=False, index=False, mode='a')
test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

print("test_set: ", len(test_set))
print("test_loader: ", len(test_loader))

model = torch.load('model.pth')            
model.eval()
print("Starting Testing...")
# test_corrects = 0
eps_list = list(np.arange(0, 0.101, 0.001))
for eps in eps_list:
    if (eps * 1000) % 10 == 0:
        # 儲存原始樣本
        x.to_csv("eps={:1.2f}_adv_concat.csv".format(eps), header=False, index=False, mode='w')
    test_acc = 0.0
    # test_loss = 0.0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels) 
        model.zero_grad()
        batch_loss.backward() 
        data_grad = inputs.grad.data
        adv = inputs + eps * data_grad.sign()
        if (eps * 1000) % 10 == 0:
            # 儲存對抗樣本
            pd.DataFrame(adv.cpu().detach().numpy()).to_csv("eps={:1.2f}_adv_concat.csv".format(eps), header=False, index=False, mode='a')
        adv_outputs = model(adv)
        # _, test_pred = torch.max(outputs, 0)
        # test_corrects += (test_pred.cpu() == labels.cpu()).sum().item()
        # test_acc = float(test_corrects/len(test_set))
        test_acc += acc(adv_outputs, labels).item()
        # test_loss += batch_loss.item()
        # if i % 10 == 9:  # 每 10 個 batch 輸出一次
        #     print('[{:03d}/{:03d}] Acc: {:3.6f} loss: {:3.6f}'.format(i+1, len(test_loader), acc(outputs, labels).item(), batch_loss.item()))
    test_acc = test_acc/len(test_loader)
    print('eps={:1.3f} Acc: {:1.6f}'.format(eps, test_acc))
    with open('eps_effect.txt', 'a') as f:
        f.write('eps={:1.3f} Acc: {:1.6f}\n'.format(eps, test_acc))