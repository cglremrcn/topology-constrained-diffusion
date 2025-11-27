import torch
from torch.utils.data import Dataset
import numpy as np

from dataset_generator import generate_circle_data, generate_disk_data

class PointCloudDataset(Dataset):
    def __init__(self, dataset_size = 1000,n_points = 200):
        """
        Args:
            dataset_size(int): Total examples in the dataset
            n_points(int): Number of points in each example
        """
        self.dataset_size = dataset_size
        self.n_points = n_points
        self.data = []
        self.labels = []

        print(f"{dataset_size} examples are being generated")

        for i in range(dataset_size):
            if np.random.rand() > 0.5:
                
                points = generate_circle_data(n_points)
                label = 1
            else:
                points = generate_disk_data(n_points)
                label = 0
  
            self.data.append(torch.tensor(points,dtype = torch.float32))
            self.labels.append(torch.tensor(label,dtype = torch.long))
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    dataset = PointCloudDataset(dataset_size = 10, n_points = 500)

    sample_data, sample_label = dataset[0]
    
    print("shape",sample_data.shape)
    print(f"label: {sample_label} (0: disk, 1: circle)")
    print("type",sample_data.type())

