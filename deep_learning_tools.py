import torch
import torch.nn.functional as functional

import torch
import torch.nn as nn

# Step 1: Generate a random tensor
# Assuming a batch of 10 images, 3 channels (e.g., RGB), 32x32 pixels
random_tensor = torch.randn(3, 3)
random_tensor[0] = torch.tensor([1, 2, 3])
random_tensor[1] = torch.tensor([4, 5, 6])
random_tensor[2] = torch.tensor([7, 8, 9])

# Step 2: Define the BatchNorm layer
# The number of features should match the number of channels in the input tensor
batch_norm = nn.BatchNorm1d(num_features=3)

# Step 3: Apply batch normalization
normalized_tensor = batch_norm(random_tensor)

print("Original Tensor:")
print(random_tensor)

print("\nNormalized Tensor:")
print(normalized_tensor)
