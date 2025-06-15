import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import onnx
from onnxsim import simplify
from torchvision import models, transforms
import time
from torchsummary import summary
from thop import profile

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Load pretrained ResNet18 model
resnet18 = models.resnet18(pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
resnet18.eval().to(device)

# Print summary for parameters and MACs
summary(resnet18, (3, 224, 224))  # For ResNet18, image size is (3, 224, 224)

# Calculate MACs (Multiply-Accumulate Operations) and Parameters
input_tensor = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(resnet18, inputs=(input_tensor, ))

print(f"Total Number of Parameters: {params / 1e6:.2f} million")
print(f"Total Number of MACs: {macs / 1e9:.2f} billion")

# Image URIs
uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
]

# Batch sizes to test
batch_sizes = [1, 32, 64, 128]
latencies = []
throughputs = []

# Inference loop
for batch_size in batch_sizes:
    x_uri_bs = [uris[0] for _ in range(batch_size)]
    batch = torch.cat([
        utils.prepare_input_from_uri(uri) for uri in x_uri_bs
    ]).to(device)

    # Measure inference time
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):  # Run 100 times to reduce measurement noise
            output = torch.nn.functional.softmax(resnet18(batch), dim=1)
        end_time = time.time()

    avg_latency = (end_time - start_time) / 100
    throughput = batch_size / avg_latency

    latencies.append(avg_latency)
    throughputs.append(throughput)

    print(f"Batch Size: {batch_size}, Latency: {avg_latency:.6f} sec, Throughput: {throughput:.2f} images/sec")


# Bar Graph: Latency and Throughput
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar for Latency
ax1.bar(batch_sizes, latencies, width=10, color='b', alpha=0.6, label='Latency (s)')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Latency (seconds)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Secondary Y-axis for Throughput
ax2 = ax1.twinx()
ax2.plot(batch_sizes, throughputs, 'r-o', label='Throughput (images/sec)')
ax2.set_ylabel('Throughput (images/sec)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Title and Grid
plt.title('Batch Size vs Latency & Throughput (ResNet18)')
fig.tight_layout()
plt.grid(True)
plt.show()

# Export the model to ONNX format for Netron visualization
onnx_path = "resnet18.onnx"
dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Dummy input for export
torch.onnx.export(resnet18, dummy_input, onnx_path, opset_version=11)

# Simplify the ONNX model
onnx_model = onnx.load(onnx_path)
simplified_model, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated."
onnx.save(simplified_model, "resnet18_simplified.onnx")

print("ONNX model saved as 'resnet18_simplified.onnx'. Open it in Netron for visualization.")
