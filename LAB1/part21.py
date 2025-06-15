import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Training settings
n_epochs = 1  
batch_size = 32  # Fixed batch size
log_interval = 100  

learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load MNIST dataset
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

# CNN Model with Variable Filter Size
class SimpleCNN(nn.Module):
    def __init__(self, kernel_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fixed fully connected layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(network, optimizer, epoch):
    network.train()
    total_training_time = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_end_time = time.time()
        total_training_time += (batch_end_time - batch_start_time)
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return total_training_time

# Testing function
def test(network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Function to measure inference time
def measure_inference_time(network):
    with torch.no_grad():
        test_iterator = iter(test_loader)
        data, _ = next(test_iterator)
        single_batch_start = time.time()
        for _ in range(1000):  
            _ = network(data)
        single_batch_end = time.time()
        single_batch_inf_time = (single_batch_end - single_batch_start) / 1000
        print(f"Single Batch Inference time is {single_batch_inf_time:.8f} seconds for a batch size of {test_loader.batch_size}")
        return single_batch_inf_time

# Experiment: Train CNN with 3x3, 5x5, and 7x7 filters
for kernel_size in [3, 5, 7]:
    print(f"\nTraining CNN with {kernel_size}x{kernel_size} filters...\n")
    network = SimpleCNN(kernel_size)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    total_training_time = 0
    for epoch in range(1, n_epochs + 1):
        total_training_time += train(network, optimizer, epoch)

    accuracy = test(network)
    inference_time = measure_inference_time(network)

    # Calculate total parameters and MACs
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total Parameters: {total_params}")

    # MACs estimation: Multiply total parameters by 2 (for convolutional layers, it's different but this is an approximation)
    total_macs = total_params * 2
    print(f"Total MACs: {total_macs}")

    # Store data for analysis
    print(f"\nResults for CNN with {kernel_size}x{kernel_size} filters:")
    print(f"Training Time: {total_training_time:.6f} seconds")
    print(f"Inference Time: {inference_time:.8f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Parameters: {total_params}")
    print(f"Total MACs: {total_macs}\n")
