import torch
import h5py
from transition_data import OfflineTransitionDataset

def split_dataset(input_path, train_path, val_path, train_fraction=0.8):
    """
    Load an OfflineTransitionDataset from disk, split it into train and validation sets,
    and save the splits back to disk as HDF5 files.

    Args:
        input_path (str): Path to the input HDF5 dataset file.
        train_path (str): Path where to save the training set HDF5 file.
        val_path (str): Path where to save the validation set HDF5 file.
        train_fraction (float): Fraction of data to use for training (default 0.8).
    """
    # Load the full dataset into memory
    dataset = OfflineTransitionDataset(input_path, fraction=1.0, load=True)
    
    n = len(dataset)
    train_size = int(train_fraction * n)
    val_size = n - train_size
    
    print(f"Total samples: {n}")
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Split the tensors
    train_data = {
        "o": dataset.observation[:train_size].cpu().numpy(),
        "a": dataset.action[:train_size].cpu().numpy(),
        "r": dataset.reward[:train_size].cpu().numpy(),
        "oprime": dataset.next_observation[:train_size].cpu().numpy(),
        "terminated": dataset.terminated[:train_size].cpu().numpy(),
        "truncated": dataset.truncated[:train_size].cpu().numpy(),
        "state": dataset.state[:train_size].cpu().numpy(),
        "state_prime": dataset.next_state[:train_size].cpu().numpy(),
    }
    
    val_data = {
        "o": dataset.observation[train_size:].cpu().numpy(),
        "a": dataset.action[train_size:].cpu().numpy(),
        "r": dataset.reward[train_size:].cpu().numpy(),
        "oprime": dataset.next_observation[train_size:].cpu().numpy(),
        "terminated": dataset.terminated[train_size:].cpu().numpy(),
        "truncated": dataset.truncated[train_size:].cpu().numpy(),
        "state": dataset.state[train_size:].cpu().numpy(),
        "state_prime": dataset.next_state[train_size:].cpu().numpy(),
    }
    
    # Save training set
    with h5py.File(train_path, "w") as f:
        for key, data in train_data.items():
            f.create_dataset(key, data=data)
    print(f"Training set saved to {train_path}")
    
    # Save validation set
    with h5py.File(val_path, "w") as f:
        for key, data in val_data.items():
            f.create_dataset(key, data=data)
    print(f"Validation set saved to {val_path}")

if __name__ == "__main__":
    # Example usage - replace with your actual paths
    input_path = "/path/to/data/green-exploration.h5"
    train_path = "/path/to/data/green-exploration-train.h5"
    val_path = "/path/to/data/green-exploration-val.h5"
    
    split_dataset(input_path, train_path, val_path)