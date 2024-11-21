import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from dataloader import GreenFunctionDataset
from architecture import GVGG
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

# Ensure TensorFlow detects the GPU
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
print("Devices detected:", devices)

# Load the dataset with a single file
dataset = GreenFunctionDataset(file_paths=["ggft/bin/data20241121/gft_16_4.bin"], batch_size=32, shuffle=True)


# Load data from the single file
data, gf = dataset._parse_raw("ggft/bin/data20241121/gft_16_4.bin")

# Split the data
train_data, test_data, train_gf, test_gf = train_test_split(data, gf, train_size=0.8, random_state=42)

# Create TensorFlow datasets from the split data
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_gf)).batch(32).shuffle(100)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_gf)).batch(32)



# Define the autoencoder
class Autoencoder(tf.keras.Model):
    def __init__(self, args, is_training=True):
        super(Autoencoder, self).__init__()
        self.network = GVGG(is_training=is_training, args=args)

    def call(self, inputs, training=False):
        # In an autoencoder, input and output shapes are the same
        return self.network.get_pred(inputs, training)

# Training function
def train_autoencoder(train_dataset, test_dataset, args, save_path="best_autoencoder.h5"):
    # Define the model
    model = Autoencoder(args)
    
    # Define the optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    loss_fn = tf.keras.losses.mean_squared_error
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Training loop
        for data, _ in train_dataset:  # No labels for autoencoder
            with tf.GradientTape() as tape:
                reconstructed = model(data, training=True)
                loss = loss_fn(data, reconstructed)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Testing loop
        test_loss = 0
        num_batches = 0
        for data, _ in test_dataset:  # No labels for autoencoderx
            reconstructed = model(data, training=False)
            test_loss += loss_fn(data, reconstructed).numpy()
            num_batches += 1
        test_loss /= num_batches

        print(f"Test Loss: {test_loss:.4f}")
        if test_loss < best_loss:
            best_loss = test_loss
            model.save_weights(save_path)

    print(f"Training complete. Best test loss: {best_loss:.4f}")

# Arguments for training
class Args:
    def __init__(self):
        self.group = 'S4'   # Group configuration
        self.epochs = 10

args = Args()
# Train the autoencoder
train_autoencoder(train_dataset, test_dataset, args, save_path="best_autoencoder.h5")
