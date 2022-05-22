#import libraries
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight') #setting the background color for the plot


#upload the file
from google.colab import files
uploaded = files.upload()


#read the file
import io

def get_aa_onehot_map(all_aa):
    aa_map = {aa: i for i, aa in enumerate(all_aa)}
    aa_onehot = np.zeros((20, 20))
    aa_onehot[np.arange(20), np.array(range(20))] = 1
    return {aa: aa_onehot[i] for i, aa in enumerate(all_aa)}
  
def get_ss_map(total_ss):
    if total_ss == 3:
        ss_map = {"E": 0, "H": 1, "T": 2}
        ss_map_r = {0: "E", 1: "H", 2: "T"}
        return ss_map, ss_map_r
    elif total_ss == 8:
        ss_map = {"E": 0, "H": 1, "T": 2, "C": 3, "S": 4, "B": 5, "G": 6, "I": 7 }
        ss_map_r = {0: "E", 1: "H", 2: "T", 3: "C", 4: "S", 5: "B", 6: "G", 7: "I"}
        return ss_map, ss_map_r  

def get_sequences(uploaded, total_ss, csv_file=True):
    all_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_onehot_map = get_aa_onehot_map(all_aa)
    ss_map, ss_map_r = get_ss_map(total_ss)
    names = []
    seq_list = []
    ss_list = []
    if csv_file:
        
        pdb_df = pd.read_csv("database.csv") #read the dataset
        seq = "".join(pdb_df["seq"].tolist())
        if total_ss == 3:
            ss = "".join(pdb_df["sst3"].tolist())
        else:
            ss = "".join(pdb_df["sst8"].tolist())

        actual_seq = ""
        actual_ss = ""
        for j in range(len(seq)):
            if seq[j] in all_aa and ss[j] in ss_map:
                actual_seq += seq[j]
                actual_ss += ss[j]

        return all_aa, aa_onehot_map, ss_map, ss_map_r, actual_seq, actual_ss

total_ss = 8
all_aa, aa_onehot_map, ss_map, ss_map_r, actual_seq, actual_ss = get_sequences("database.csv", total_ss) #specify the dataset file


# Plotting protein secondary structure
ss_count = {}
for ss in actual_ss:
    if ss not in ss_count:
        ss_count[ss] = 0
    else:
        ss_count[ss] += 1
print([round(c / sum(ss_count.values()), 4) for c in ss_count.values()])
mc = np.array(sorted(ss_count.items(), key=lambda x: x[1], reverse=True))
_, ax = plt.subplots(figsize=(10, 5))
ax.bar(mc[:, 0], np.array(mc[:, 1], dtype=int))
plt.title("Protein secondary structure distribution")
plt.show()


#Plotting amino acid distribution
aa_count = {}
for aa in actual_seq:
    if aa not in aa_count:
        aa_count[aa] = 0
    else:
        aa_count[aa] += 1
mc = np.array(sorted(aa_count.items(), key=lambda x: x[1], reverse=True))
_, ax = plt.subplots(figsize=(10, 5))
plt.bar(mc[:, 0], np.array(mc[:, 1], dtype=int))
plt.ylim(ymin=100)
plt.title("Amino acid distribution")
plt.show()


#one-hot encoding visualization
class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, all_seq, all_ss, w_size, aa_onehot, ss_onehot_map):
        self.all_seq = all_seq
        self.all_ss = all_ss
        self.w_size = w_size
        self.aa_onehot = aa_onehot
        self.ss_onehot_map = ss_onehot_map

    def __len__(self):
        return len(self.all_seq) - self.w_size + 1

    def __getitem__(self, idx):
        seq_onehot = [self.aa_onehot[self.all_seq[aa]] for aa in range(idx, idx + self.w_size)]
        ss_onehot = self.ss_onehot_map[self.all_ss[idx + int(self.w_size / 2)]]
        return np.array(seq_onehot)[None], np.array(ss_onehot)
    
    
#using modules conv2D, BatchNorm2D and ReLu
    class Net(nn.Module):
    def __init__(self, height, out_size):
        super(Net, self).__init__()
        self.height = height
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn_2d_a = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn_2d_b = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        '''
        self.rnn = nn.RNN(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.gru = nn.GRU(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        '''

        self.fc1 = nn.Linear(64 * int((self.height - 4) / 2) * 8, 128)
        self.bn_1d_a = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_1d_b = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, out_size)

    def forward(self, x):
        x = self.relu(self.bn_2d_a(self.conv1(x)))
        x = self.relu(self.bn_2d_b(self.conv2(x)))
        
        x = self.dropout1(self.pool(x))

        # x, h_n = self.rnn(x.view(-1, int((self.height - 4) / 2), 8), None)
        # x, h_n = self.gru(x.view(-1, int((self.height - 4) / 2), 8), None)
        # x, (h_n, h_c) = self.lstm(x.view(-1, int((self.height - 4) / 2), 8), None)
        # x = torch.flatten(x[:, -1, :], 1)
        # x = x.view(128, -1)
        
        x = torch.flatten(x, 1)
        x = self.relu(self.bn_1d_a(self.fc1(x)))
        x = self.relu(self.bn_1d_b(self.fc2(x)))
        
        x = self.fc3(self.dropout2(x))
        
        return F.log_softmax(x, dim=1)
    

#define the training 
    def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.type(torch.LongTensor).cuda())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 5000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    return running_loss / batch_idx


#define the test
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target.type(torch.LongTensor).cuda(), reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct / len(test_loader.dataset)


#running the epochs
# Running the epoch
w_size = 21
batch_size = 128
total_ss = 3

all_aa, aa_onehot_map, ss_map, ss_map_r, actual_seq, actual_ss = get_sequences("database.csv", total_ss)
seq_train, seq_test, ss_train, ss_test = train_test_split(actual_seq, actual_ss, test_size=0.25, shuffle=False)

train_dataset = ProteinDataset(seq_train, ss_train, w_size, aa_onehot_map, ss_map)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = ProteinDataset(seq_test, ss_test, w_size, aa_onehot_map, ss_map)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(w_size, total_ss).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

loss_values = []
accuracy_values = []
test(model, device, test_loader)
for epoch in range(10):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    loss_values.append(train_loss)

    test_acc = test(model, device, test_loader)
    accuracy_values.append(test_acc)

    scheduler.step()

torch.save(model.state_dict(), "model_ss3.pth")


#plotting the train loss and test accuracy
_, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(1, 11).astype(int), loss_values, label="train loss")
ax.plot(np.arange(1, 11).astype(int), accuracy_values, label="test accuracy")
plt.xlabel('Epochs')
plt.ylabel('Loss and Acc')
plt.title('Plot of Train Loss and Test Accuracy')
plt.legend()
plt.show()


#summary of the model
summary(model, input_size=(1, 21, 20))


#Visualization of alpha helix and beta sheet
w_size = 21
batch_size = 512
datasets = ProteinDataset(seq_train, ss_train, w_size, aa_onehot_map, ss_map)
loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True)
next_batch_0 = next(iter(loader))
next_batch_1 = next(iter(loader))
next_batch_2 = next(iter(loader))
next_batch_3 = next(iter(loader))
next_batch_4 = next(iter(loader))
next_batch_5 = next(iter(loader))

all_next_batch_x = torch.cat((
    next_batch_0[0], next_batch_1[0], next_batch_2[0], 
    next_batch_3[0], next_batch_4[0], next_batch_5[0]
))
all_next_batch_y = torch.cat((
    next_batch_0[1], next_batch_1[1], next_batch_2[1], 
    next_batch_3[1], next_batch_4[1], next_batch_5[1]
))
all_E_table = all_next_batch_x[all_next_batch_y == 0]
all_H_table = all_next_batch_x[all_next_batch_y == 1]
all_T_table = all_next_batch_x[all_next_batch_y == 2]
print("E: {}, H: {}, T: {}".format(len(all_E_table), len(all_H_table), len(all_T_table)))

min_table = min(len(all_E_table), len(all_H_table), len(all_T_table))
all_E_table = all_E_table[:min_table]
all_H_table = all_H_table[:min_table]
all_T_table = all_T_table[:min_table]
all_E_summed = np.squeeze(all_E_table.sum(axis=0), axis=0)
all_H_summed = np.squeeze(all_H_table.sum(axis=0), axis=0)
all_T_summed = np.squeeze(all_T_table.sum(axis=0), axis=0)
print("E: {}, H: {}, T: {}".format(all_E_summed.max(), all_H_summed.max(), all_T_summed.max()))

vmax = max(all_E_summed.max(), all_H_summed.max(), all_T_summed.max())
vmax = min(all_E_summed.max(), all_H_summed.max(), all_T_summed.max())

fig, axs = plt.subplots(1, 3, figsize=(40, 10)) # 10, 30

sns.heatmap(all_E_summed, cmap="Greys", xticklabels=all_aa, ax=axs[0], vmin=0, vmax=vmax)
axs[0].set_title("Visualization of {} E Batchs".format(len(all_E_table)), fontsize=18)
axs[0].set_xlabel("Amino Acids", fontsize=18)
axs[0].set_ylabel("Window Index", fontsize=18)
axs[0].invert_yaxis()

sns.heatmap(all_H_summed, cmap="Greys", xticklabels=all_aa, ax=axs[1], vmin=0, vmax=vmax)
axs[1].set_title("Visualization of {} H Batchs".format(len(all_H_table)), fontsize=18)
axs[1].set_xlabel("Amino Acids", fontsize=18)
axs[1].set_ylabel("Window Index", fontsize=18)
axs[1].invert_yaxis()

sns.heatmap(all_T_summed, cmap="Greys", xticklabels=all_aa, ax=axs[2], vmin=0, vmax=vmax)
axs[2].set_title("Visualization of {} T Batchs".format(len(all_T_table)), fontsize=18)
axs[2].set_xlabel("Amino Acids", fontsize=18)
axs[2].set_ylabel("Window Index", fontsize=18)
axs[2].invert_yaxis()

plt.show()


#Visualization of H Batch
w_size = 21
batch_size = 1
datasets = ProteinDataset(seq_train, ss_train, w_size, aa_onehot_map, ss_map)
loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True)
temp_batch = next(iter(loader))
viz_batch = temp_batch[0]
print(viz_batch.shape)
summed = np.squeeze(viz_batch.sum(axis=0), axis=0)
print(summed.shape)
plt.figure(figsize=(10,7))
ax = sns.heatmap(summed, cmap="Greys", xticklabels=all_aa)
plt.title("Visualization of {} Batch".format(ss_map_r[temp_batch[1].item()]), fontsize=18)
plt.xlabel("Amino Acids", fontsize=18)
plt.ylabel("Window Index", fontsize=18)
plt.gca().invert_yaxis()
plt.show()