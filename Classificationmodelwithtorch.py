import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
from torch import mode, nn,optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
# make 100 samples
n_samples=1000
# create circles
x,y=make_circles(n_samples,noise=0.03,random_state=42)
print(x[:5])
print(y[:5])
print(y)
circls=pd.DataFrame({"x1":x[:,0],
                     "x2":x[:,1],
                     "label":y})
print(circls.head(10))
plt.scatter(x=x[:,0],y=x[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.show()
print(x.shape,y.shape)
x_sample=x[0]
y_sample=y[0]
print(x_sample,y_sample)
# turn data imot tensor and create train and test splits
x_tensor=torch.from_numpy(x).type(torch.float)
print(x_tensor)
print(x_tensor.dtype)
y_tensor=torch.from_numpy(y).type(torch.float)
print(y_tensor.dtype)
torch.manual_seed(42)
#split data into training and test sets
x_train,x_test,y_train,y_test=train_test_split(x_tensor,y_tensor,test_size=0.2,random_state=42)
print(len(x_train),len(x_test),len(y_train),len(x_test))
# let's build a model 
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
class CircleModelv0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=10)
        self.layer2=nn.Linear(in_features=10,out_features=1)
        
    def forward(self,x):
        return self.layer2(self.layer1(x))
model=CircleModelv0().to(device)
print(model)
model1=nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features=5,out_features=1)
)
print(x_test)
# with torch.no_grad():
#  untrained_preds=model1(x_train)
# print(model.state_dict(),model1.state_dict())
# print(f"length of predictions:{len(untrained_preds)}")
# print(f"lenngth of test sample:{len(x_test)}")
# print(f"first 10 predictions:{untrained_preds[:10]}")
# print(f"first 10 labels:{y[:,10]}")
# loss=nn.BCEWithLogitsLoss()
optimizer=optim.SGD(params=model.parameters(),lr=0.1)
# print(loss,optimizer)
print(model.state_dict())
#calculate the accuracy
# train the model
# froward pass
# optimizer zero grad
# loss backward backpropagation 
# optimizer step gradient desent
model.eval()
with torch.inference_mode():
  y_logits=model(x_test.to(device))
  print(y_logits)
print(y_test[:5])
y_pred_prob=torch.round(torch.sigmoid(x_test.to(device)[:5]))
print(y_pred_prob)
print(torch.round(y_pred_prob))
# print(torch.eq(y_pred_prob.squeeze()))
print(torch.squeeze(y_pred_prob))
# training the model 
# put the data to target device
# loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100
epochs=10000
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# move data to device
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
for epoch in range(epochs):
    model.train()
    # forward
    logits = model(x_train)
    preds = torch.round(torch.sigmoid(logits))
    # loss
    loss = loss_fn(logits, y_train.unsqueeze(1))
    acc = accuracy_fn(y_train, preds)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
     # evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test.unsqueeze(1))
        test_acc = accuracy_fn(y_test, test_preds)
    if epoch % 100 == 0:
        print(
            f"epoch {epoch} | "
            f"loss {loss:.4f} | "
            f"acc {acc:.2f}% | "
            f"test_acc {test_acc:.2f}%"
        )
print(model.state_dict())
# visualize the decision boundary
# to do so we are going to import a function called plot_decision_boundary
def plot_decision_boundary(model, X, y):
    model.eval()

    # Move data to CPU and numpy
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # Set min and max values
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    # Create grid tensor
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits)

    Z = probs.reshape(xx.shape).cpu().numpy()

    # Plot
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()
# plot decision boundary
plot_decision_boundary(model, x_train, y_train)
plot_decision_boundary(model, x_test, y_test)
# add more layers to the model 
class CircleModelv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=10)
        self.layer2=nn.Linear(in_features=10,out_features=10)
        self.layer3=nn.Linear(in_features=10,out_features=1)
        
    def forward(self,x):
        return self.layer3(self.layer2(self.layer1(x)))
torch.cuda.manual_seed(42)
model2=CircleModelv1().to(device)
print(model2)
optimizer2=optim.SGD(params=model2.parameters(),lr=0.1)
loss_fn2=nn.BCEWithLogitsLoss()
epochs=1000
# with torch.inference_mode():