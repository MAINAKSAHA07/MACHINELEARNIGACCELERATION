import os
import numpy as np
from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)
from transformers.data.processors.squad import SquadV1Processor
from transformers import squad_convert_examples_to_features
from torch.utils.data import DataLoader
import torch
import time

# Create cache directory
cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Download SQuAD dataset if not already present
predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
predict_file = os.path.join(cache_dir, "dev-v1.1.json")
if not os.path.exists(predict_file):
    import wget
    print("Downloading SQuAD dataset...")
    wget.download(predict_file_url, predict_file)
    print("Download complete.")

# BERT model and configuration
model_name_or_path = "bert-base-cased"
max_seq_length = 128
doc_stride = 128
max_query_length = 64
total_samples = 100
device = torch.device("cpu")

# Load pretrained model and tokenizer
config = BertConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
model = BertForQuestionAnswering.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

# Compute total parameters and MACs
def count_parameters_and_macs(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_macs = sum(p.numel() * 2 for p in model.parameters() if p.requires_grad)  # Approximation
    return total_params, total_macs

total_params, total_macs = count_parameters_and_macs(model)
print(f"Total Parameters: {total_params}")
print(f"Total MACs: {total_macs}")

# Load examples from SQuAD
processor = SquadV1Processor()
examples = processor.get_dev_examples(None, filename=predict_file)

features, dataset = squad_convert_examples_to_features(
    examples=examples[:total_samples],
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False,
    return_dataset='pt',
    threads=1
)

# Batch sizes to experiment with
batch_sizes = [1, 8, 16, 32]

# Set model to evaluation mode
model.eval()
model.to(device)

# Measure inference latency for different batch sizes
results = []
for batch_size in batch_sizes:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Warm-up iterations to stabilize measurement
    for _ in range(10):
        for batch in data_loader:
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'token_type_ids': batch[2].to(device)
            }
            model(**inputs)
            break

    # Measure latency
    start_time = time.time()
    for _ in range(50):  # 50 iterations for stable measurement
        for batch in data_loader:
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'token_type_ids': batch[2].to(device)
            }
            model(**inputs)
            break
    end_time = time.time()

    latency = (end_time - start_time) / 50  # Average latency per iteration
    throughput = 1 / latency if latency > 0 else 0  # Inferences per second
    results.append((batch_size, latency, throughput))
    print(f"Batch size: {batch_size}, Latency: {latency:.6f} seconds, Throughput: {throughput:.2f} inferences/sec")

# Generate bar graph
import matplotlib.pyplot as plt

batch_sizes, latencies, throughputs = zip(*results)

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Latency (seconds)', color=color)
ax1.bar(batch_sizes, latencies, color=color, alpha=0.6, label='Latency')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Throughput (inferences/sec)', color=color)
ax2.plot(batch_sizes, throughputs, color=color, marker='o', linestyle='dashed', label='Throughput')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('BERT Inference Latency vs Throughput')
plt.show()
