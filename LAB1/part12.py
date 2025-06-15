import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Hyperparameters
n_epochs = 1  
batch_size = 64  
learning_rate = 0.01
momentum = 0.5

torch.manual_seed(1)

# Load MNIST Dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

# Define MLP Model with variable hidden sizes
class SimpleMLP(nn.Module):
    def __init__(self, hidden_layers):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_size = 28 * 28  # Input size (flattened MNIST image)
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        layers.append(nn.Linear(prev_size, 10))  # Output layer (10 classes)
        layers.append(nn.LogSoftmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten input
        return self.model(x)

# Function to count total number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Function to estimate the number of Multiply-Accumulate Operations (MACs)
def estimate_macs(hidden_layers):
    macs = 0
    prev_size = 28 * 28  # Input size
    for layer_size in hidden_layers:
        macs += prev_size * layer_size  # Multiply-Accumulate operations per layer
        prev_size = layer_size
    macs += prev_size * 10  # Last layer
    return macs * 2  # Each MAC operation has two operations (mult & add)

# Function to train the model and record metrics
def train_and_evaluate(hidden_layers, label):
    print(f"\nTraining {label} model with hidden layers: {hidden_layers}")
    network = SimpleMLP(hidden_layers)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # Measure training time
    start_time = time.time()
    network.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time

    # Measure inference time
    network.eval()
    inference_start = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            network(data)  # Forward pass
    inference_time = (time.time() - inference_start) / len(test_loader)

    # Evaluate accuracy
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    total_params = count_parameters(network)
    total_macs = estimate_macs(hidden_layers)

    print(f"{label} Model Results:")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Inference Time: {inference_time:.6f} seconds per batch")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Parameters: {total_params}")
    print(f"Total MACs: {total_macs}\n")

    return hidden_layers, training_time, inference_time, accuracy, total_params

# Define network variations
base_layers = [256, 256, 256]
small_layers = [128, 128, 128]
large_layers = [1024, 1024, 1024]

# Train and record results
results = []
results.append(train_and_evaluate(base_layers, "Base"))
results.append(train_and_evaluate(small_layers, "Smaller"))
results.append(train_and_evaluate(large_layers, "Larger"))

# Print results in table format
print("\nFinal Comparison Table:")
print("| Network Layer Neurons | Training Time (s) | Inference Time (s) | Accuracy (%) | Total Parameters |")
print("|----------------------|------------------|------------------|--------------|----------------|")
for res in results:
    print(f"| {res[0]} | {res[1]:.4f} | {res[2]:.6f} | {res[3]:.2f} | {res[4]} |")
