from pathlib import Path
from sys import platlibdir
import torch 
from torch import nn,optim
from torchaudio.datasets import tedlium
import librosa
from torch.utils.data import DataLoader,Dataset
from torchaudio.transforms import AmplitudeToDB
from torchaudio.models import wav2vec2_base
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from tqdm.auto import tqdm
from skimage.transform import resize
device='cuda' if torch.cuda.is_available() else 'cpu'
data_df=pd.read_csv('files_paths.csv')
print(data_df.head())
print(data_df['Class'].unique())
print(data_df['Class'].unique().shape)
print(data_df['Class'].value_counts())
print(data_df['Class'].argmax())
path_dir=Path('Dataset/')
plt.figure(figsize=(10,7))
plt.pie(data_df['Class'].value_counts(),labels=data_df['Class'].value_counts().index,autopct='%1.1f')
plt.title('class distribution')
plt.show()
# data_df['Filpath']=path_dir+data_df['FilPath'].str[1:]
# train_data=data_df.sample(frac=.70,random_state=42)
# test_data=data_df.drop(train_data.index)
# val_data=test_data.sample(frac=0.5,random_state=42)
# test_data=val_data.drop(val_data.index)
label_encoder = LabelEncoder()
data_df['Class'] = label_encoder.fit_transform(data_df['Class'])
train_data = data_df.sample(frac=0.70, random_state=42)
test_data = data_df.drop(train_data.index)
val_data = test_data.sample(frac=0.5, random_state=42)
test_data = test_data.drop(val_data.index)
print(f"Train shape:{train_data.shape}")
print(f"Test data:{test_data.shape}")
print(f"Valid data :{val_data.shape}")
# class Customdataset(Dataset):
#     def __init__(self,dataframe,):
#         self.dataframe=dataframe
#         self.labels=torch.Tensor(dataframe['Class']).type(torch.LongTensor)
#         self.audios=[torch.Tensor(self.get_spectogram(path)).type(torch.FloatTensor) for path in dataframe['FilePath']]
#     def __len__(self):
#         return self.dataframe.shape[0]
#     def __getitem__(self, index):
#         img_path=self.dataframe.iloc[index,0]
#         label=torch.Tensor(self.labels[index])
#         audio=self.audios[index].unsqueeze(0)
#         return audio,label
#     def get_spectrogram(self,filepath):
#         sample_rate=22050
#         duration=5
#         img_height=128
#         img_width=256
#         signal,sr=librosa.load(filepath,sr=sample_rate,duration=duration)
#         spec=librosa.feature.melspectrogram(y=signal,sr=sample_rate,n_fft=2040,hop_length=512,n_mels=120)
#         spec_db=librosa.power_to_db(spec,ref=np.max)
#         spec_resize=librosa.util.fix_length(spec_db,size=(duration*sample_rate)//512+1)
#         spec_resize=resize(spec_resize,(img_height,img_width),anti_aliasing=True)
#         return spec_resize
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index):
        filepath = self.dataframe.loc[index, 'FilePath']
        label = torch.tensor(
            self.dataframe.loc[index, 'Class'],
            dtype=torch.long
        )
        spec = self.get_spectrogram(filepath)
        return spec, label
    def get_spectrogram(self, filepath):
        sample_rate = 22050
        duration = 5
        img_height = 128
        img_width = 256
        signal, _ = librosa.load(
            filepath,
            sr=sample_rate,
            duration=duration
        )
        spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_resized = resize(
            spec_db,
            (img_height, img_width),
            anti_aliasing=True
        )
        spec_resized = (spec_resized - spec_resized.mean()) / (spec_resized.std() + 1e-6)
        spec_tensor = torch.tensor(
            spec_resized,
            dtype=torch.float32
        ).unsqueeze(0)

        return spec_tensor

train_Data=CustomDataset(dataframe=train_data)
test_Data=CustomDataset(dataframe=test_data)
val_Data=CustomDataset(dataframe=val_data)
sample_spec = train_Data.get_spectrogram('C:\\Users\\91956\\OneDrive\\Desktop\\Project\\RNN\\Dataset\\AbdulBari_Althubaity\\abdulbari_001.wav')
print(f"Sample spectrogram shape: {sample_spec.shape}")
lr=1e-4
BATCH_SIZE=16
epochs=20
train_dataloader=DataLoader(train_Data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader=DataLoader(test_Data,batch_size=BATCH_SIZE,shuffle=True)
val_dataloader=DataLoader(val_Data,batch_size=BATCH_SIZE,shuffle=True)
sample_batch = next(iter(train_dataloader))
print(f"Batch shape: {sample_batch[0].shape}")
class AudioRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(AudioRNN,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        x=x.squeeze(1).permute(0,2,1)
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        out,_=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out
# class NeuralNetwork(nn.Module):
#     def __init__(self,input_shape,hidden_units,output_shape):
#         super().__init__()
#         self.conv1=nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1)
#         self.conv2=nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units*2,kernel_size=3,stride=1,padding=1)
#         self.conv3=nn.Conv2d(in_channels=hidden_units*2,out_channels=hidden_units*4,kernel_size=3,stride=1,padding=1)
#         self.pooling=nn.MaxPool2d(kernel_size=2,stride=2)
#         self.relu=nn.ReLU()
#         self.flatten=nn.Flatten()
#         self.fc1=nn.Linear(in_features=(hidden_units*4*16*32),out_features=hidden_units*256)
#         self.fc2=nn.Linear(in_features=hidden_units*256,out_features=hidden_units*64)
#         self.fc3=nn.Linear(in_features=hidden_units*64,out_features=hidden_units*32)
#         self.output=nn.Linear(in_features=hidden_units*32,out_features=output_shape)
#         self.dropout=nn.Dropout(0.3)
#     def forward(self,x):
#         x=self.conv1(x)
#         x=self.relu(x)
#         x=self.pooling(x)
#         x=self.conv2(x)
#         x=self.relu(x)
#         x=self.pooling(x)
#         x=self.conv3(x)
#         x=self.relu(x)
#         x=self.pooling(x)
#         # x=x.view(x.size(0),-1)
#         x=self.flatten(x)
#         x=self.fc1(x)
#         x=self.relu(x)
#         x=self.dropout(x)
#         x=self.fc2(x)
#         x=self.relu(x)
#         x=self.dropout(x)
#         x=self.fc3(x)
#         x=self.relu(x)
#         x=self.dropout(x)
#         x=self.output(x)
#         return x
class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape, hidden_units, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_units)
        self.bn2 = nn.BatchNorm2d(hidden_units*2)
        self.bn3 = nn.BatchNorm2d(hidden_units*4)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_units*4*4*4, 256)
        self.fc2 = nn.Linear(256, output_shape)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
num_classes = len(data_df['Class'].unique())
BATCH_SIZE = 16
lr = 1e-3
epochs = 20
model_cnn=NeuralNetwork(input_shape=1,hidden_units=16,output_shape=num_classes).to(device)
model_rnn=AudioRNN(input_size=128,hidden_size=256,num_layers=2,num_classes=num_classes).to(device)
print(model_cnn.state_dict())
print(model_rnn.state_dict())
loss_fn = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=lr)
optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=lr)

# ------------------ Training Loop ------------------
train_loss_cnn_list, train_acc_cnn_list = [], []
train_loss_rnn_list, train_acc_rnn_list = [], []

test_loss_cnn_list, test_acc_cnn_list = [], []
test_loss_rnn_list, test_acc_rnn_list = [], []

for epoch in tqdm(range(epochs), desc="Epochs"):
    # ------------------ Training ------------------
    model_cnn.train()
    model_rnn.train()

    total_loss_cnn = 0
    total_acc_cnn = 0
    total_loss_rnn = 0
    total_acc_rnn = 0

    for data, targets in train_dataloader:
        data = data.to(device)
        targets = targets.to(device)

        # ----- CNN -----
        optimizer_cnn.zero_grad()
        outputs_cnn = model_cnn(data)
        loss_cnn = loss_fn(outputs_cnn, targets)
        loss_cnn.backward()
        optimizer_cnn.step()

        total_loss_cnn += loss_cnn.item()
        preds_cnn = outputs_cnn.argmax(dim=1)
        total_acc_cnn += (preds_cnn == targets).sum().item()

        # ----- RNN -----
        optimizer_rnn.zero_grad()
        outputs_rnn = model_rnn(data)
        loss_rnn = loss_fn(outputs_rnn, targets)
        loss_rnn.backward()
        optimizer_rnn.step()

        total_loss_rnn += loss_rnn.item()
        preds_rnn = outputs_rnn.argmax(dim=1)
        total_acc_rnn += (preds_rnn == targets).sum().item()

    train_loss_cnn_list.append(total_loss_cnn/len(train_dataloader))
    train_acc_cnn_list.append(total_acc_cnn/len(train_Data))

    train_loss_rnn_list.append(total_loss_rnn/len(train_dataloader))
    train_acc_rnn_list.append(total_acc_rnn/len(train_Data))

    # ------------------ Testing ------------------
    model_cnn.eval()
    model_rnn.eval()
    total_loss_cnn_test, total_acc_cnn_test = 0, 0
    total_loss_rnn_test, total_acc_rnn_test = 0, 0

    with torch.no_grad():
        for data, targets in test_dataloader:
            data = data.to(device)
            targets = targets.to(device)

            # CNN
            outputs_cnn = model_cnn(data)
            loss_cnn = loss_fn(outputs_cnn, targets)
            total_loss_cnn_test += loss_cnn.item()
            preds_cnn = outputs_cnn.argmax(dim=1)
            total_acc_cnn_test += (preds_cnn == targets).sum().item()

            # RNN
            outputs_rnn = model_rnn(data)
            loss_rnn = loss_fn(outputs_rnn, targets)
            total_loss_rnn_test += loss_rnn.item()
            preds_rnn = outputs_rnn.argmax(dim=1)
            total_acc_rnn_test += (preds_rnn == targets).sum().item()

    test_loss_cnn_list.append(total_loss_cnn_test/len(test_dataloader))
    test_acc_cnn_list.append(total_acc_cnn_test/len(test_Data))

    test_loss_rnn_list.append(total_loss_rnn_test/len(test_dataloader))
    test_acc_rnn_list.append(total_acc_rnn_test/len(test_Data))

    print(f"Epoch {epoch+1}/{epochs} | "
          f"CNN Loss: {train_loss_cnn_list[-1]:.4f}, CNN Acc: {train_acc_cnn_list[-1]:.4f} | "
          f"RNN Loss: {train_loss_rnn_list[-1]:.4f}, RNN Acc: {train_acc_rnn_list[-1]:.4f} | "
          f"Test CNN Acc: {test_acc_cnn_list[-1]:.4f}, Test RNN Acc: {test_acc_rnn_list[-1]:.4f}")

# ------------------ Plot Loss & Accuracy ------------------
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.plot(train_loss_cnn_list, label='Train Loss CNN')
plt.plot(test_loss_cnn_list, label='Test Loss CNN')
plt.plot(train_loss_rnn_list, label='Train Loss RNN')
plt.plot(test_loss_rnn_list, label='Test Loss RNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_cnn_list, label='Train Acc CNN')
plt.plot(test_acc_cnn_list, label='Test Acc CNN')
plt.plot(train_acc_rnn_list, label='Train Acc RNN')
plt.plot(test_acc_rnn_list, label='Test Acc RNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()
