
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from utils import *





def add_missing_columns(d, columns):
    missing_col = set(columns) - set(d.columns)
    for col in missing_col:
        d[col] = 0


def fix_columns(d, columns):
    add_missing_columns(d, columns)
    assert (set(columns) - set(d.columns) == set())
    d = d[columns]
    return d


def data_process(df, model):
    df.replace(" ?", pd.NaT, inplace=True)
    if model == 'train':
        df.replace(" >50K", 1, inplace=True)
        df.replace(" <=50K", 0, inplace=True)
    if model == 'test':
        df.replace(" >50K.", 1, inplace=True)
        df.replace(" <=50K.", 0, inplace=True)

    df.replace(" Male", 1, inplace=True)
    df.replace(" Female",0,inplace=True)


    trans = {'workclass': df['workclass'].mode()[0], 'occupation': df['occupation'].mode()[0],
             'native-country': df['native-country'].mode()[0]}
    df.fillna(trans, inplace=True)
    target = df["income"]
    sensitive = df['sex']

    df.drop('sex',axis=1,inplace=True)
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('capital-gain', axis=1, inplace=True)
    df.drop('capital-loss', axis=1, inplace=True)

    df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
    df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income']

    dataset = pd.concat([df[df_int_col], pd.get_dummies(df[df_object_col])], axis=1)

    return target,sensitive, dataset


class Adult_data(Dataset):
    def __init__(self, model):
        super(Adult_data, self).__init__()
        self.model = model
        adult_train_path = r'E:\Data\Adult\adult.data'
        adult_test_path = r'E:\Data\Adult\adult.test'
        df_train = pd.read_csv(adult_train_path, header=None,
                               names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                      'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                      'hours-per-week', 'native-country', 'income'])
        df_test = pd.read_csv(adult_test_path, header=None, skiprows=1,
                              names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                     'hours-per-week', 'native-country', 'income'])

        train_target,train_sensitive, train_dataset = data_process(df_train, 'train')
        test_target,test_sensitive, test_dataset = data_process(df_test, 'test')

        #         进行独热编码对齐
        test_dataset = fix_columns(test_dataset, train_dataset.columns)
        #         print(df["income"])
        train_dataset = train_dataset.apply(lambda x: (x - x.mean()) / x.std())
        test_dataset = test_dataset.apply(lambda x: (x - x.mean()) / x.std())
        #         print(train_dataset['native-country_ Holand-Netherlands'])

        train_target, test_target = np.array(train_target), np.array(test_target)
        train_sensitive, test_sensitive = np.array(train_sensitive), np.array(test_sensitive)
        train_dataset, test_dataset = np.array(train_dataset, dtype=np.float32), np.array(test_dataset,
                                                                                          dtype=np.float32)
        if model == 'test':
            isnan = np.isnan(test_dataset)
            test_dataset[np.where(isnan)] = 0.0
        #             print(test_dataset[ : , 75])

        if model == 'test':
            self.target = torch.tensor(test_target, dtype=torch.int64)
            self.sensitive = torch.tensor(test_sensitive, dtype=torch.int64)
            self.dataset = torch.FloatTensor(test_dataset)
        else:
            #           前百分之八十的数据作为训练集，其余作为验证集
            if model == 'train':
                self.target = torch.tensor(train_target, dtype=torch.int64)[: int(len(train_dataset) * 0.8)]
                self.sensitive = torch.tensor(train_sensitive, dtype=torch.int64)[: int(len(train_dataset) * 0.8)]
                self.dataset = torch.FloatTensor(train_dataset)[: int(len(train_target) * 0.8)]
            else:
                self.target = torch.tensor(train_target, dtype=torch.int64)[int(len(train_target) * 0.8):]
                self.sensitive = torch.tensor(train_sensitive, dtype=torch.int64)[int(len(train_dataset) * 0.8):]
                self.dataset = torch.FloatTensor(train_dataset)[int(len(train_dataset) * 0.8):]
        #print(self.dataset.shape, self.target.dtype)

    def __getitem__(self, item):
        return self.dataset[item], self.target[item],self.sensitive[item]

    def __len__(self):
        return len(self.dataset)


train_dataset = Adult_data(model='train')
val_dataset = Adult_data(model='val')
test_dataset = Adult_data(model='test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

class Adult_Model(nn.Module) :
    def __init__(self) :
        super(Adult_Model, self).__init__()
        self.net = nn.Sequential(nn.Linear(100, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 2)
                                )
    def forward(self, x) :
        out = self.net(x)
        return out #F.softmax(out)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = Adult_Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = F.binary_cross_entropy_with_logits#nn.CrossEntropyLoss()
max_epoch = 50# 30
classes = [' <=50K', ' >50K']
mse_loss = 1000000
os.makedirs('Models', exist_ok=True)
#writer = SummaryWriter(log_dir='logs')

train_flag=0
if train_flag:
    for epoch in range(max_epoch):

        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for x, label, sensitive in train_loader:
            x, label,sensitive = x.to(device), label.to(device), sensitive.to(device)
            a=label*sensitive
            b=label*(1-sensitive)
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
            two_target = torch.cat([a,b], dim=1)

            optimizer.zero_grad()

            out = model(x)
            out_0=out[:,0].unsqueeze(0)
            out_1 = out[:, -1].unsqueeze(0)
            a = sensitive.unsqueeze(1)
            b=1-a

            mask = torch.cat((a, b), dim=1) == 1
            pred = torch.masked_select(out, mask) >0
            pred=pred.long()

            # if sensitive==0:
            #     #loss = criterion(out_0, label)
            #     _, pred = torch.max(out_0, 1)
            # else:
            #     #loss = criterion(out_1, label)
            #     _, pred = torch.max(out_1, 1)

            output = torch.cat([out_0, out_1], dim=1)

            loss=criterion(output, two_target.float())
            train_loss += loss.item()
            loss.backward()
            num_correct = (pred == label).sum().item()
            acc = num_correct / x.shape[0]
            train_acc += acc
            optimizer.step()
        if epoch%10==9:
            print(        f'epoch : {epoch + 1}, train_loss : {train_loss / len(train_loader.dataset)}, train_acc : {train_acc / len(train_loader)}')

        # with torch.no_grad():
        #     total_loss = []
        #     model.eval()
        #     for x, label,sensitive in val_loader:
        #         x, label = x.to(device), label.to(device)
        #     out = model(x)
        #     loss = criterion(out, label)
        #     total_loss.append(loss.item())
        #
        #     val_loss = sum(total_loss) / len(total_loss)
        #
        #     if val_loss < mse_loss:
        #         mse_loss = val_loss
    torch.save(model.state_dict(), 'Models/Adult_domain_50.pth')

model = Adult_Model().to(device)
ckpt = torch.load('Models/Adult_domain_50.pth', map_location='cpu')
model.load_state_dict(ckpt)

test_loss = 0.0
test_acc = 0.0
model.eval()

group_set = torch.tensor([], dtype=torch.long)
target_set = torch.tensor([], dtype=torch.long)
pred_set = torch.tensor([], dtype=torch.long)


for x, label,sensitive in test_loader:
    x, label,sensitive = x.to(device), label.to(device), sensitive.to(device)


    #loss = criterion(out, label)
    #test_loss += loss.item()
    #_, pred = torch.max(out, dim=1)
    #result.append(pred.detach())

    out = model(x)
    out_0 = out[:, 0].unsqueeze(0)
    out_1 = out[:, -1].unsqueeze(0)
    a = sensitive.unsqueeze(1)
    b = 1 - a

    mask = torch.cat((a, b), dim=1) == 1
    pred = torch.masked_select(out, mask) > 0
    pred = pred.long()
    group_set = torch.cat((group_set, sensitive.cpu()))
    target_set = torch.cat((target_set, label.cpu()))
    pred_set = torch.cat((pred_set, pred.cpu()))
    num_correct = (pred == label).sum().item()
    acc = num_correct / x.shape[0]
    test_acc += acc

print(f' test_acc : {test_acc / len(test_loader)}')

acc,TPR_abs,TNR_abs,DP=cal_rate(y=target_set,prediction=pred_set,z=group_set)
print(acc,TPR_abs,TNR_abs,DP)

