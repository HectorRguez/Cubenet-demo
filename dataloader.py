import tensorflow as tf
import numpy as np

class GreenFunctionDataset:
    def __init__(self, file_paths, batch_size, shuffle=True, N=16, dtype=np.float32):
        """
        Initialize the dataset loader.

        Args:
            file_paths: List of file paths for the dataset.
            batch_size: Batch size for training/testing.
            shuffle: Whether to shuffle the dataset.
            N: Number of Green's function blocks (default: 16).
            dtype: Data type for the binary file (default: np.float64).
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

    def _load_sample(self, file_path):
        """
        Wrap _parse_raw to load data and return tensors.

        Args:
            file_path: File path to the binary file.

        Returns:
            Tuple of dielectric constants and Green's function as tensors.
        """
        data, gf = self._parse_raw(file_path.numpy().decode("utf-8"))
        return tf.convert_to_tensor(data, dtype=tf.float32), tf.convert_to_tensor(gf, dtype=tf.float32)

    def _tf_parse(self, file_path):
        """
        TensorFlow wrapper for _load_sample.

        Args:
            file_path: File path tensor.

        Returns:
            Tuple of dielectric constants and Green's function tensors.
        """
        data, gf = tf.py_function(func=self._load_sample, inp=[file_path], Tout=[tf.float32, tf.float32])
        return data, gf

    def get_dataset(self):
        """
        Create the TensorFlow dataset pipeline.

        Returns:
            A tf.data.Dataset object.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.file_paths))
        dataset = dataset.map(self._tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
