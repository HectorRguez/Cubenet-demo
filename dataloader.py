import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GreenFunctionDataset(Dataset):
    def __init__(self, file_paths, batch_size, shuffle=True, N=16, dtype=np.float32):
        """
        Initialize the dataset loader.

        Args:
            file_paths: List of file paths for the dataset.
            batch_size: Batch size for training/testing.
            shuffle: Whether to shuffle the dataset.
            N: Number of Green's function blocks (default: 16).
            dtype: Data type for the binary file (default: np.float32).
        """
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = N
        self.dtype = dtype

    def _parse_raw(self, address_file):
        """
        Function to read floats from a binary file.

        Args:
            address_file: File path to the binary file.

        Returns:
            data: Dielectric constants as an array.
            gf: Green's function as an array.
        """
        data = np.fromfile(address_file, dtype=self.dtype)
        n = int(data[0])
        block_width = int(data[1])
        print(n)
        print(block_width)
        blockn = n // block_width
        
        length = blockn * blockn * blockn + n * n * 6
        data = data[2:]
        
        # data = data[2:length*1000+2] # TODO Fix this problem

        data = data.reshape(-1, length)
        data, gf = np.split(data, [blockn * blockn * blockn], axis=1)
        data = data.reshape(-1, blockn, blockn, blockn, 1)
        return data, gf

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset by index.
        """
        file_path = self.file_paths[idx]
        data, gf = self._parse_raw(file_path)
        
        # Convert to torch tensors
        data = torch.tensor(data, dtype=torch.float32)
        gf = torch.tensor(gf, dtype=torch.float32)
        
        return data, gf

    def get_dataloader(self):
        """
        Create a PyTorch DataLoader.

        Returns:
            A DataLoader object.
        """
        dataset = self  # self is already a Dataset object
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

