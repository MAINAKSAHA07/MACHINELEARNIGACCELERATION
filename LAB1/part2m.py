import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

n_epochs = 1  # Number of Epochs for training
batch_size = 64 # Batch size for training and testing TODO: Modify this variable to change batch size
log_interval = 100 # This variable manages how frequently do you want to print the training loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
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

network = SimpleCNN().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_macs(model, input_tensor):
    macs = 0
    def count_mac(layer, inp, out):
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            macs += layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * out.shape[2] * out.shape[3]
        elif isinstance(layer, nn.Linear):
            macs += layer.in_features * layer.out_features
    
    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(count_mac))
    
    model(input_tensor)
    for hook in hooks:
        hook.remove()
    
    return macs


def train(epoch):
    network.train()
    total_training_time = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_start_time = time.time()
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_end_time = time.time()
        total_training_time += (batch_end_time - batch_start_time)
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())
            train_counter.append(batch_idx * batch_size + (epoch - 1) * len(train_loader.dataset))
    return total_training_time

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

total_time = 0
for epoch in range(1, n_epochs + 1):
    time_per_epoch = train(epoch)
    total_time += time_per_epoch
    test()

print(f"Total Training time: {total_time:.6f} seconds")

with torch.no_grad():
    test_iterator = iter(test_loader)
    data, target = next(test_iterator)
    data, target = data.to(device), target.to(device)
    single_batch_start = time.time()
    for _ in range(1000):
        output = network(data)
    single_batch_end = time.time()
    single_batch_inf_time = (single_batch_end - single_batch_start) / 1000
    print(f"Single Batch Inference time is {single_batch_inf_time:.8f} seconds for a batch size of {test_loader.batch_size}")

# Compute model parameters and MACs
input_tensor = torch.randn(1, 1, 28, 28).to(device)
param_count = count_parameters(network)
macs = compute_macs(network, input_tensor)

print(f"Total Parameters: {param_count}")
print(f"Total MACs: {macs}")
