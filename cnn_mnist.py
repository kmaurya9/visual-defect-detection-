import torch                                                                                                                                                 
import torch.nn as nn
from torchvision import datasets, transforms                                                                                                                 
from torch.utils.data import DataLoader    

transform = transforms.Compose([                                                                                                                        
      transforms.ToTensor(),                                                                                                                              
      transforms.Normalize((0.1307,), (0.3081,))                                                                                                          
  ])              

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)                                                                
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
                                                                                                                                                          
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)                                                                                   
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):                                                                                               
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(32 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = x.view(-1, 32 * 7 * 7)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")