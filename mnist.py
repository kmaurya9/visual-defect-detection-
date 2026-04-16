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

class MLP(nn.Module):                                                                                                                                        
    def __init__(self):                                                                                                                                      
        super().__init__()                                                                                                                                   
        self.fc1 = nn.Linear(784, 128)                                                                                                                       
        self.fc2 = nn.Linear(128, 10)                                                                                                                        
        self.relu = nn.ReLU()                                                                                                                                
                                                                                                                                                               
    def forward(self, x):                                                                                                                                  
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))                                                                                                                           
        x = self.fc2(x)
        return x      
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                                                                                        
model = MLP().to(device)                                                                                                                                     
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)                                                                                                     
criterion = nn.CrossEntropyLoss()   

                                       
for epoch in range(10):                                                                                                                                       
    model.train()                                                                                                                                            
    for images, labels in train_loader:                                                                                                                      
        images, labels = images.to(device), labels.to(device)
                                                                                                                                                               
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
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)                                                                                                               
        correct += (predicted == labels).sum().item()
        total += labels.size(0)                                                                                                                         
                                                                                                                                                          
print(f"Test Accuracy: {100 * correct / total:.2f}%")  