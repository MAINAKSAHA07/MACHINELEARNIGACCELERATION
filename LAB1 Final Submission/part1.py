import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Hyperparameters
n_epochs = 5               # Number of Epochs for training
batch_size = 64            # Batch size for training and testing
learning_rate = 0.01       # Learning rate for the optimizer
momentum = 0.5             # Momentum factor for the optimizer
log_interval = 100         # Frequency of logging training status

# Seed for reproducibility
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Data Preprocessing and Loading
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, transform=transform),
    batch_size=batch_size, shuffle=False
)

# MLP Model Definition
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

# Training and Testing Functions
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total_training_time

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

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Function to Run Different Model Configurations
def run_experiments():
    configurations = {
        "Base Model": [256, 256, 256],
        "Smaller Model": [128, 128, 128],
        "Larger Model": [1024, 1024, 1024],
        "Less Layers Model": [256, 256],
        "More Layers Model": [256, 256, 256, 256]
    }

    for label, hidden_layers in configurations.items():
        print(f"Training {label} with hidden layers: {hidden_layers}")
        network = SimpleMLP(hidden_layers=hidden_layers)
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

        total_time = 0
        for epoch in range(1, n_epochs + 1):
            time_per_epoch = train(network, optimizer, epoch)
            total_time += time_per_epoch
            test(network)

        print("Total Training time for {}: {:.4f} seconds\n".format(label, total_time))

        # Inference Timing
        with torch.no_grad():
            test_iterator = iter(test_loader)
            data, _ = next(test_iterator)
            single_batch_start = time.time()
            for _ in range(1000):
                network(data)
            single_batch_end = time.time()
            single_batch_inf_time = (single_batch_end - single_batch_start) / 1000
            print("Single Batch Inference time is {:.6f} seconds for a batch size of {}\n".format(
                single_batch_inf_time, test_loader.batch_size))


run_experiments()
