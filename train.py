import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from dataloader import GreenFunctionDataset  # Your custom dataset loader
from architecture import GVGG  # Your custom architecture

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset with a single file
dataset = GreenFunctionDataset(file_paths=["ggft/bin/data20241121/gft_16_4.bin"], batch_size=32, shuffle=True)

# Load data from the single file
data, gf = dataset._parse_raw("ggft/bin/data20241121/gft_16_4.bin")

# Split the data
train_data, test_data, train_gf, test_gf = train_test_split(data, gf, train_size=0.8, random_state=42)

# Create PyTorch datasets and dataloaders
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                               torch.tensor(train_gf, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                              torch.tensor(test_gf, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the autoencoder model class
class Autoencoder(nn.Module):
    def __init__(self, args, is_training=True):
        super(Autoencoder, self).__init__()
        self.network = GVGG(is_training=is_training, args=args)

    def forward(self, inputs):
        # In an autoencoder, input and output shapes are the same
        return self.network.get_pred(inputs, is_training=True)

# Training function
def train_autoencoder(train_loader, test_loader, args, save_path="best_autoencoder.pth"):
    # Define the model
    model = Autoencoder(args).to(device)
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Training loop
        model.train()  # Set model to training mode
        for data, _ in train_loader:  # No labels for autoencoder
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = loss_fn(reconstructed, data)
            loss.backward()
            optimizer.step()

        # Testing loop
        model.eval()  # Set model to evaluation mode
        test_loss = 0
        num_batches = 0
        with torch.no_grad():  # No gradients needed for evaluation
            for data, _ in test_loader:  # No labels for autoencoder
                data = data.to(device)
                reconstructed = model(data)
                test_loss += loss_fn(reconstructed, data).item()  # Accumulate loss
                num_batches += 1
        test_loss /= num_batches

        print(f"Test Loss: {test_loss:.4f}")
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best test loss: {best_loss:.4f}")

# Arguments for training
class Args:
    def __init__(self):
        self.group = 'S4'  # Group configuration
        self.epochs = 10

args = Args()

# Train the autoencoder
train_autoencoder(train_loader, test_loader, args, save_path="best_autoencoder.pth")
