from matplotlib import axis
import torch
from torch import nn,optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
from pathlib import Path
from PIL  import Image
import os
import shutil
from torchvision import models
from torchvision.models import vgg19_bn
from torch import xpu
from tqdm.auto import tqdm
data_path=Path('bean-leaf-lesions-classification/')
train_df=pd.read_csv("bean-leaf-lesions-classification/train.csv")
test_df=pd.read_csv("bean-leaf-lesions-classification/val.csv")
print(train_df.head())
print(train_df['category'].unique())
print(train_df.shape,test_df.shape)
print(train_df['category'].value_counts())
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.8, 1.0)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# class CustomImageDataset(Dataset):
#     def __init__(self,dataframe,transform):
#         self.dataframe=dataframe
#         self.transform=transform
#         self.labels = torch.tensor(
#                      dataframe['category'].values, dtype=torch.long)
#     def __len__(self):
#         return self.dataframe.shape[0]
#     def __getitem__(self, index):
#         img_path=self.dataframe.iloc[index,0]
#         img=Image.open(img_path).convert('RGB')
#         label=self.labels[index]
#         if self.transform:
#             img=self.transform(img)
#         return img,label
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = torch.tensor(
            dataframe['category'].values, dtype=torch.long
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = data_path/self.dataframe.iloc[index, 0]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label
train_dataset=CustomImageDataset(train_df,transform=train_transform)
val_dataset=CustomImageDataset(test_df,transform=val_transform)
n_row, n_col = 3, 3
fig, ax = plt.subplots(n_row, n_col, figsize=(6,6))
for row in range(n_row):
    for col in range(n_col):
        random_idx = np.random.randint(len(train_dataset))
        img, label = train_dataset[random_idx]
        img = img.permute(1, 2, 0)
        ax[row, col].imshow(img)
        ax[row, col].set_title(f"Label: {label.item()}")
        ax[row, col].axis("off")
plt.tight_layout()
plt.show()
train_dataloader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_dataloader=DataLoader(dataset=val_dataset,batch_size=32,shuffle=False)
model=vgg19_bn()
model1 = models.googlenet(
    weights=models.GoogLeNet_Weights.IMAGENET1K_V1,
)
print(model1.state_dict())
print(model1.fc)
print(len(train_df['category'].unique()))
model1.fc=nn.Linear(in_features=model1.fc.in_features,out_features=len(train_df['category'].unique()))
print(model1.fc)
print(model1)
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model1.parameters(),lr=3e-4)
total_loss_train_plot=[]
total_acc_train_plot=[]
total_loss_test_plot=[]
total_acc_test_plot=[]
epochs=10
for epoch in tqdm(range(epochs)):
    model1.train()
    total_train_loss=0
    total_train_acc=0
    for input,labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model1(input)
        # logits=outputs.logits
        loss=loss_fn(outputs,labels)
        total_train_loss+=loss.item()
        loss.backward()
        preds = torch.argmax(outputs, dim=1)
        train_acc = (preds == labels).sum().item()
        total_train_acc+=train_acc
        optimizer.step()
    total_loss_train_plot.append(round(total_train_loss/1000,4))
    total_acc_train_plot.append(round((total_train_acc/train_dataset.__len__())*100))
    print(f"\n Epoch {epoch+1} train loss {round(total_train_loss/1000,4)} train_acc {round((total_train_acc/train_dataset.__len__())*100)}%")
    with torch.no_grad():
        model1.eval()
        total_test_loss=0
        total_test_acc=0
        for input,labels in test_dataloader:
            outputs=model1(input)
            # logits=outputs.logits
            loss=loss_fn(outputs,labels)
            total_test_loss+=loss.item()
            preds=torch.argmax(outputs,dim=1)
            test_acc=(preds==labels).sum().item()
            total_test_acc+=test_acc
    total_loss_test_plot.append(round(total_test_loss/1000,4))
    total_acc_test_plot.append(round((total_test_acc/val_dataset.__len__())*100))
    print(f"\n Epoch {epoch+1} test loss {round(total_test_loss/1000,4)} test_acc {round((total_test_acc/val_dataset.__len__())*100)}%")
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7))
ax[0].plot(total_loss_train_plot,label='training_loss')
ax[0].plot(total_loss_test_plot,label='validationloss')
ax[0].set_title("Training and validations loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[1].plot(total_acc_train_plot,label='training_acc')
ax[1].plot(total_acc_test_plot,label='validation_acc')
ax[1].set_title("Training and validations acc")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("accuracy")
ax[1].legend()
plt.show()













