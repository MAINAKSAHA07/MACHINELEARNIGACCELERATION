import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Define the MLP class with batch normalization and dropout
class MLP(nn.Module):
    def __init__(self, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        input_size = 28 * 28
        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.BatchNorm1d(h))  # Batch Normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout to prevent overfitting
            input_size = h
        layers.append(nn.Linear(input_size, 10))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# Function to train and evaluate models
def train_and_evaluate(hidden_layers, batch_size=64, epochs=10, lr=0.001):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_train = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_train
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    inference_start = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    inference_time = (time.time() - inference_start) / len(test_loader)
    accuracy = 100. * correct / total
    
    total_params = sum(p.numel() for p in model.parameters())
    total_macs = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2  # Approximate MACs
    
    print(f"Hidden Layers: {hidden_layers}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Inference Time: {inference_time:.6f} seconds per batch")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Parameters: {total_params}")
    print(f"Total MACs: {total_macs}\n")
    
# Train different model variations
print("Training Base Model")
train_and_evaluate([256, 256, 256])
print("Training Less Layers Model")
train_and_evaluate([256, 256])
print("Training More Layers Model")
train_and_evaluate([256, 256, 256, 256])
