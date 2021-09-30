import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import Variable
from torchvision import transforms

import pandas as pd
import os
import random

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

path = './dataset/0818(1000)/'
np.random.seed(10)
size = 20000
kp_list = []
idx = 0
train_test_ratio = 0.1

for i in range(2, 6):
    for j in range(1, 251):
        if j % 50 == 0:
            print(i, j)
        kappa = np.array(pd.read_csv(path+'{}_{}_k.csv'.format(i, j), header=None))
        phi = np.array(pd.read_csv(path+'{}_{}_p.csv'.format(i, j), header=None))
        s = np.array(pd.read_csv(path+'{}_{}_s.csv'.format(i, j), header=None))
        temp = np.zeros((s.shape[0], s.shape[1]+1))
        rnd = np.random.choice(len(kappa), size, replace=False)

        for k in range(len(s)):
            if s[k][4] == 10:
                s[k][4] = 0
            elif s[k][4] == 50:
                s[k][4] = 1
            elif s[k][4] == 100:
                s[k][4] = 2
            elif s[k][4] == 200:
                s[k][4] = 3
            elif s[k][4] == 300:
                s[k][4] = 4
            elif s[k][4] == 400:
                s[k][4] = 5
            elif s[k][4] == 500:
                s[k][4] = 6
            
            temp[k] = np.append(s[k], np.array([idx]))

        np.random.shuffle(temp)
        train_temp, test_temp = np.split(temp, [int(train_test_ratio*len(temp))])
        
        if idx == 0:
            train_set = train_temp
            test_set = test_temp
            # state = temp
        else:
            train_set = np.append(train_set, train_temp, axis=0)
            test_set = np.append(test_set, test_temp, axis=0)
            # state = np.append(state, temp, axis=0)

        idx += 1

        kp = []

        for l in rnd:
            tmp = []
            k = kappa[l]
            p = phi[l]
            k = list(k)
            k.extend(list(p))
            kp.append(k)

        kp = torch.tensor(kp, dtype=torch.float).to(device)
        kp = kp.view((1, 1, size, 3))
        kp_list.append(kp)

np.random.shuffle(train_set)
print(train_set)
print(train_set.shape, test_set.shape)
# np.random.shuffle(state)

class Custom_Dataset(Dataset):
    def __init__(self, state):
        self.x = state[:, :4]
        self.target = state[:, 4]
        self.idx = state[:, 5]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index]).float()
        y = torch.tensor(self.target[index]).float()
        idx = torch.tensor(self.idx[index]).int()
        
        return x, y, idx

class NN(torch.nn.Module):
    def __init__(self, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, output):
        super(NN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, (1, 3)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.hidden1 = nn.Linear(size, h1)
        self.hidden2 = nn.Linear(h1, h2)
        self.hidden3 = nn.Linear(h2, h3)
        self.hidden4 = nn.Linear(h3, h4)
        # self.hidden5 = nn.Linear(h4, h5)
        # self.hidden6 = nn.Linear(h5, h6)
        # self.hidden7 = nn.Linear(h6, h7)

        self.hidden8 = nn.Linear(h4+4, h8)
        self.hidden9 = nn.Linear(h8, h9)
        self.hidden10 = nn.Linear(h9, h10)
        # self.hidden11 = nn.Linear(h10, h11)
        self.predict = nn.Linear(h10, output)

        # nn.init.xavier_uniform_(self.conv.weight)
        # nn.init.xavier_uniform_(self.hidden1.weight)
        # nn.init.xavier_uniform_(self.hidden2.weight)
        # nn.init.xavier_uniform_(self.hidden3.weight)
        # nn.init.xavier_uniform_(self.predict.weight)


    def forward(self, x1, x2):
        x2 = x2.view((-1,4))

        x1 = self.conv(x1)
        x1 = x1.view((-1,size))

        x1 = F.relu(self.hidden1(x1))
        x1 = F.relu(self.hidden2(x1))
        x1 = F.relu(self.hidden3(x1))
        x1 = F.relu(self.hidden4(x1))
        # x1 = F.relu(self.hidden5(x1))
        # x1 = F.relu(self.hidden6(x1))
        # x1 = F.relu(self.hidden7(x1))

        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = F.relu(self.hidden10(x))
        # x = F.relu(self.hidden11(x))
        x = self.predict(x)
        return x

h1 = 1024
h2 = 512
h3 = 128
h4 = 10
h5 = 512
h6 = 128
h7 = 10
h8 = 512
h9 = 512
h10 = 512
h11 = 512
model = NN(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, output=7).to(device)
loss_func = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr = 0)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=0.0001, step_size_up=1, step_size_down=199, gamma=0.95, mode='exp_range' , cycle_momentum=False)

def test(model, mode, verbose=False):
    model.eval()

    correct = 0
    
    if mode == "val":
        loader = val_loader
        dataset = val_dataset
    elif mode == "test":
        loader = test_loader
        dataset = test_dataset

    with torch.no_grad():
        
        for x, y, idx in loader:
            error = []
            truth = []
            x, y = x.float().to(device), y.to(device)
            for n, index in enumerate(idx):
                if n == 0:
                    kps = kp_list[index]
                else:
                    kps = torch.cat([kps, kp_list[index]], dim=0)
            predict = model(kps, x)

            predict = torch.argmax(predict, 1)

            if verbose:
                print("Predict : ", predict)
                print("Target : ", y)
                # for idx in torch.nonzero(predict-y):
                #     error.append(predict[idx].item())
                #     truth.append(int(y[idx].item()))
                # print("Predict : ", error)
                # print("TRUTH   : ",truth)
                # print('\n')
            
            correct += (predict == y).sum().item()
    
    accuracy = 100 * float(correct) / len(dataset)
    print('\nTest accuracy : {:3f}%'.format(accuracy))

    return accuracy

from sklearn.model_selection import KFold

kfold = KFold(n_splits=4, shuffle=True, random_state=0)

batch_size = 32

stop = False

best = []

for eps in range(10000):

    for fold_index, (t, v) in enumerate(kfold.split(train_set)):

        torch.cuda.empty_cache()
        train_data = train_set[t]
        val_data = train_set[v]

        train_dataset = Custom_Dataset(train_data)
        val_dataset = Custom_Dataset(val_data)

        train_loader = DataLoader(train_dataset,batch_size=batch_size)
        val_loader = DataLoader(val_dataset,batch_size=batch_size)

        val_acc_max = 0

        for epoch in range(5):
            model.train()
            print('\n{}_{}fold_{}th epoch'.format(eps+1, fold_index+1, epoch+1))
            # print(list(model.parameters()))
            current_lr = lr_scheduler.get_last_lr()
            print("\nCurrent learning rate : ", current_lr)
                
            for i, data in enumerate(train_loader):
                x, y, idx = data
                x, y = x.float().to(device), y.to(device)
                for n, index in enumerate(idx):
                    if n == 0:
                        kps = kp_list[index]
                    else:
                        kps = torch.cat([kps, kp_list[index]], dim=0)
                # print(kps.shape)        

                predict = model(kps, x)

                loss = loss_func(predict, y.long())
                
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                if i % 200 == 0:
                    print('#'*100)
                    # print('Target : {}'.format(y))
                    # print('Predict : {}'.format(predict))
                    print('Trian loss : {}'.format(loss))
            
            val_acc = test(model, 'val')

            lr_scheduler.step()
            
            if val_acc_max < val_acc:
                val_acc_max = val_acc
                models = [model, val_acc_max]

            if val_acc >= 98.5:
                stop = True

            if stop:
                torch.save(model, f'./result/c1_{h1}_{h2}_{h3}_{h4}_{h5}_{h6}_{h7}_{h8}_{h9}_{h10}({val_acc}).pth')
                break
        if stop:
            break
                
        best.append(models)
    if stop:
        break
        