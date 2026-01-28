from pathlib import Path
import test
import torch
from torch import nn,optim
import matplotlib.pyplot as plt
import torchvision 
import numpy as np
import random
from PIL import Image
from  torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.datasets import StanfordCars
import pandas as pd
from torchvision.datasets import ImageFolder
data_path=Path('Bone_Fracture_Binary_Classification/')
from torchvision import transforms
from pathlib import Path
from PIL import Image
from pathlib import Path
from PIL import Image

# def clean_dataset(root_dir):
#     removed = 0
#     for img_path in Path(root_dir).rglob("*"):
#         if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
#             try:
#                 with Image.open(img_path) as img:
#                     img.verify()
#             except Exception:
#                 print("Removing:", img_path)
#                 img_path.unlink()
#                 removed += 1

#     print(f"\nTotal removed images: {removed}")

# clean_dataset("Bone_Fracture_Binary_Classification/test/Fracture")

# def remove_corrupted_images(root_dir):
#     bad_files = []
#     for img_path in Path(root_dir).rglob("*.jpg"):
#         try:
#             with Image.open(img_path) as img:
#                 img.verify()
#         except Exception:
#             bad_files.append(img_path)

#     print(f"Found {len(bad_files)} corrupted images")
#     for p in bad_files:
#         print("Removing:", p)
#         p.unlink()

# remove_corrupted_images("Bone_Fracture_Binary_Classification")
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            # skip bad image by moving to next
            return self.__getitem__((index + 1) % len(self))
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),    
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
train_data=SafeImageFolder(root=data_path/"train",transform=data_transform)
test_data=SafeImageFolder(root=data_path/"test",transform=data_transform)
val_data=SafeImageFolder(root=data_path/"val",transform=data_transform)
print(f"train data:{train_data}")
print(f"Classes:{train_data.classes}")
print(f"classes idx:{train_data.class_to_idx}")
class_names=train_data.classes
num_classes=len(class_names)
print(num_classes)
image_path_list=list(data_path.glob("*/*/*.jpg")) # glob meaning steak all together
print(f"print all image path list :{image_path_list}")
random_images_path=random.choice(image_path_list)
print(random_images_path)
# get the image class from the path (the images class is the name of dir where the image stored)
images_class=random_images_path.parent.stem
print(images_class)
img=Image.open(random_images_path)
# 5. print  metadata
print(f"Random images path:{random_images_path}")
print(f"Image class:{images_class}")
print(f"Image height:{img.height}")
print(f"Image width:{img.width}")
plt.imshow(img,cmap="gray")
plt.axis(False)
plt.title(f"Image class:{images_class}")
plt.show()
data_transform=transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
def display_random_images(dataset:torch.utils.data.Dataset,classes:list[str]=None,n:int=10,display_shape:bool=True,seed:int=None):
     # 2.adjust display if n is too high 
     if n>10:
         n=10
         display_shape=False
         print("For display purposes,n should not be be larger than 10,setting to 10")
    # set the seed 
     if seed:
       random.seed(seed)
#   get the random samples
     random_sample_idx=random.sample(range(len(dataset)),k=n)
     #5
     plt.figure(figsize=(16,8))

     for i ,tar_sample in enumerate(random_sample_idx):
         tar_img,tar_label=dataset[tar_sample][0],dataset[tar_sample][1]
     # adjust tensor dim for ploting 
         tar_image_adjust=tar_img.permute(1,2,0)
         plt.subplot(1,n,i+1)
         plt.imshow(tar_image_adjust)
         plt.axis("off")
         if classes:
             title=f"class:{classes[tar_label]}"
             if display_shape:
                 title=title+f"\nshape:{tar_image_adjust.shape}"
             plt.title(title)
     plt.show()
display_random_images(dataset=train_data,n=5,classes=class_names,seed=None)
print(data_transform(img))
print(data_transform(img).shape)
BATCH_SIZE=32
class FractureDetection(nn.Module):
    def  __init__(self,input_shape=int,hidden_units=int,output_shape=int):
        super().__init__()
        self.conv_block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # default stride value is same as kernal_size
        )
        self.conv_block2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # default stride value is same as kernal_size
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*56*56,out_features=output_shape)
        )
    def forward(self,x):
       x=self.conv_block1(x)
    #    print(x.shape)
       x=self.conv_block2(x)
       x=self.classifier(x)
       return x
Modelv0=FractureDetection(input_shape=3,hidden_units=10,output_shape=len(train_data.classes))
print(Modelv0.state_dict())
print(Modelv0.parameters())
device='cuda' if torch.cuda.is_available() else 'cpu'
# class_names=train_data.classes
# def count_classes(dataset, class_name):
#     return len(list(Path.glob(dataset, f"{class_name}/*")))
# def create_class_counts_df(dataset, class_names):
#     counts = {'class':[], 'count': []}
#     for class_name in class_names:
#         counts['class'].append(class_name)
#         counts['count'].append(count_classes(dataset, class_name))
#     return pd.DataFrame(counts)

# def plot_class_distribution(df, title, palette="viridis"):
#     sns.barplot(x='class', y='count', data=df, palette=palette)
#     plt.title(title)
#     plt.xlabel('Class')
#     plt.ylabel('Count')
#     plt.show()
# train_df = create_class_counts_df(train_data, class_names)
# plot_class_distribution(train_df, 'Class Distribution in Training Set', palette='rocket')
#training loop 
epochs=1000
train_dataloader=DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_dataloader=DataLoader(dataset=test_data,batch_size=32,shuffle=False)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(Modelv0.parameters(), lr=1e-3)
for epoch in tqdm(range(epochs)):
    Modelv0.train()
    train_loss, train_acc = 0, 0

    for batch_idx, (X, y) in enumerate(train_dataloader):
        # X, y = X.to(device), y.to(device)

        logits = Modelv0(X).squeeze(dim=1)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (logits.argmax(dim=1) == y).float().sum().item()
    if epoch%10==0:
        print(f"epochs:{epoch}  loss={loss} train_loss={(train_loss/len(train_data))*100} accaracy={(train_acc/len(train_data))*100}")
    train_loss /= len(train_dataloader)
    train_acc  /= len(train_dataloader)
def test_loop(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            test_loss += loss.item()
            preds = logits.argmax(dim=1)
            test_acc += (preds == y).float().mean().item()

    test_loss /= len(dataloader)
    test_acc  /= len(dataloader)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
    return test_loss, test_acc
test_loss, test_acc = test_loop(Modelv0, test_dataloader, loss_fn, device)
def predict_image(model, image_path, transform, class_names, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return class_names[pred_idx], confidence
pred_class, confidence = predict_image(
    model=Modelv0,
    image_path=random_images_path,
    transform=data_transform,
    class_names=class_names,
    device=device
)

print(f"Prediction: {pred_class}")
print(f"Confidence: {confidence*100:.2f}%")
