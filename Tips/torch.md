# Pytorch ABC

### A. Prepare Datasets and Dataloaders

```python
torch.utils.data.Dataset
torch.utils.data.DataLoader
```

```python
# Example 
training_data = datasets.FashionMNIST(
                              root='.', 
                              train=True, 
                              download='True', 
                              transform=ToTensor()
)

from torch.utils.data import Dataset

img, label = training_data[id]
```

+ Implement a Dataset class

```python
### The images are stored in a directory /img_dir; the labels are stored in annotations.csv
import os
import pandas as pd
import torchvision.io as tvio

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = tvio.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
```

+ Use a DataLoader

  Reason: we want to pass samples in "mini batches" while training a model

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Iterate through the dataset. 
train_features, train_labels = next(iter(train_dataloader))

img = train_features[id].squeeze()
plt.imshow(img, cmap='gray')
label = train_labels[id]
```



### B. Build a Neural Network

```python
torch.nn
```

+ Implement a neural network

```python
# A subclass of nn.Module
class NeuralNetwork(nn.Module):
  	def __init__(self):
    		super(NeuralNetwork, self).__init__()
    		self.flatten = nn.Flatten()
    		self.linear_relu_stack = nn.Sequential(
        													nn.Linear(28*28, 512),
        													nn.ReLU(),
        													nn.Linear(512, 512),
        													nn.ReLU(),
        													nn.Linear(512, 10),
        													nn.ReLU(),
        )
    # implements the operations on input data in the forward method
    def forward(self, X):
      	x = self.flatten(X)
        logits = self.linear_relu_stack(x)
        return logits
```



### C. Optimize It

+ Loss and Optimizer

```python
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

+ Automatic differentiation

```python
# Compute the gradient
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

loss.backward()
w.grad
b.grad
'''
We can only perform gradient calculations using *backward* function once on a given graph, for performance reasons.
'''
```

```python
'''
We can stop tracking computations when we only want to do forward computations
'''
with torch.no_grad():
  	z = torch.matmul(x, w) + b
```

+ Loop

  Each epoch consists of two parts:

  + Train loop
  + Validation loop

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad() # reset the gradient of model parameters
        loss.backward()       # differentiation
        optimizer.step()			# adjust the parameters gradient

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

+ Train it

```python
epochs = 5
for t in range(epochs):
  	train_loop(traing_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
```

