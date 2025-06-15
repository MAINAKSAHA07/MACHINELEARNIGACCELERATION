import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Configuration
n_epochs = 1
batch_size = 32
learning_rate = 0.01
momentum = 0.5
log_interval = 100
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loaders
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# Part 1: Base CNN
class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Part 2: CNN with Different Filter Sizes
class FilterSizeCNN(nn.Module):
    def __init__(self, kernel_size):
        super(FilterSizeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Part 3: CNN with Different Number of Layers
class VariableLayersCNN(nn.Module):
    def __init__(self, num_layers):
        super(VariableLayersCNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
            in_channels = 32
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (28 // (2 ** num_layers)) ** 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Simple MLP Model
class SimpleMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_layers=[256, 256, 256], output_size=10):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_size = input_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.LogSoftmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input
        return self.model(x)

network = SimpleMLP()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []

# Training and Testing Functions
def train(model, optimizer, epoch):
    model.train()
    total_training_time = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        start_time = time.time()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_training_time += time.time() - start_time
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    return total_training_time

def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def measure_inference_time(model):
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        start_time = time.time()
        for _ in range(1000):
            model(data)
        avg_time = (time.time() - start_time) / 1000
        return avg_time

# Execution
models = [
    (BaseCNN(), "Base CNN"),
    *((FilterSizeCNN(k), f"Filter Size {k}x{k}") for k in [3, 5, 7]),
    *((VariableLayersCNN(l), f"{l} Convolution Layers") for l in [1, 2, 4]),
    (network, "Simple MLP")
]

for model, description in models:
    print(f"\nTraining {description}")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    total_training_time = 0
    for epoch in range(1, n_epochs + 1):
        total_training_time += train(model, optimizer, epoch)
    accuracy = test(model)
    inference_time = measure_inference_time(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_macs = total_params * 2  # Approximate MACs

    print(f"Results: Training Time: {total_training_time:.6f}s, Inference Time: {inference_time:.8f}s, "
          f"Accuracy: {accuracy:.2f}%, Total Parameters: {total_params}, Total MACs: {total_macs}")
