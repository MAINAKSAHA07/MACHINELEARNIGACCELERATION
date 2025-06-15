import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

n_epochs = 1  # Number of Epochs for training
batch_size = 32 # Batch size for training and testing
log_interval = 100 

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
    def __init__(self, num_conv_layers=2):
        super(SimpleCNN, self).__init__()
        self.convs = nn.ModuleList()
        in_channels = 1
        out_channels = 32
        
        for i in range(num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2))
            in_channels = out_channels
            out_channels *= 2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute the size of the flattened feature map dynamically
        self.feature_map_size = self._get_feature_map_size(num_conv_layers)
        
        self.fc1 = nn.Linear(self.feature_map_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_feature_map_size(self, num_conv_layers):
        """
        Calculates the feature map size dynamically based on the number of conv layers.
        Assumes input image is 28x28.
        """
        size = 28  # MNIST input size (28x28)
        for _ in range(num_conv_layers):
            size = size // 2  # Each pooling operation halves the size
        return (size * size) * (32 * (2 ** (num_conv_layers - 1)))  # Channels x (H x W)

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Base model with default convolution layers
base_network = SimpleCNN(num_conv_layers=2)
less_layers_network = SimpleCNN(num_conv_layers=1)  # Fewer layers
more_layers_network = SimpleCNN(num_conv_layers=4)  # More layers

optimizer = optim.SGD(base_network.parameters(), lr=learning_rate, momentum=momentum)

def train(network, epoch):
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
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def measure_performance(network):
    total_time = 0
    for epoch in range(1, n_epochs + 1):
        time_per_epoch = train(network, epoch)
        total_time += time_per_epoch
        test(network)
    print("Total Training time: {}".format(total_time))
    
    with torch.no_grad():
        test_iterator = iter(test_loader)
        data, target = next(test_iterator)
        single_batch_start = time.time()
        for i in range(0, 1000):
            output = network(data)
        single_batch_end = time.time()
        
        single_batch_inf_time = (single_batch_end - single_batch_start) / 1000
        print("Single Batch Inference time is {} seconds for a batch size of {}".format(
            single_batch_inf_time, test_loader.batch_size))

print("Base Network Results:")
measure_performance(base_network)
print("\nNetwork with Fewer Layers Results:")
measure_performance(less_layers_network)
print("\nNetwork with More Layers Results:")
measure_performance(more_layers_network)
