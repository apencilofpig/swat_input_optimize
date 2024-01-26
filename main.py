import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from SWaT_Dataset import SWaT_Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from naive_cnn import Naive_CNN

# df = pd.read_csv('/opt/data/SWaT数据集/data_newlabel.csv', header=None)
# print(df.shape)
# df.head()

np_inputs = np.load('ieee754_inputs.npy')
np_labels = np.load('labels.npy')
np_inputs = np.transpose(np_inputs, (0, 2, 1))

# 划分训练集和剩余部分
np_inputs_train, np_inputs_remain, np_labels_train, np_labels_remain = train_test_split(np_inputs, np_labels, test_size=0.01, random_state=42)

# 划分验证集和测试集
np_inputs_valid, np_inputs_test, np_labels_valid, np_labels_test = train_test_split(np_inputs_remain, np_labels_remain, test_size=0.5, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 64
num_workers = 0
lr = 1e-4
epochs = 100

train_data= SWaT_Dataset(np_inputs_train, np_labels_train)
valid_data= SWaT_Dataset(np_inputs_valid, np_labels_valid)
test_data= SWaT_Dataset(np_inputs_test, np_labels_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = Naive_CNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(epoch, train_loader):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

def val(epoch, test_loader):       
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
    
for epoch in range(1, epochs+1):
    train(epoch, train_loader)
    val(epoch, valid_loader)