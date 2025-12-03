import torch
import torch.nn as nn

class TopologicalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x, labels):
        """
        x: [Batch, N, 2] -> Dot cloud
        labels: [Batch] -> 0 (Disk) or 1 (Ring)
        """
        
        radii = torch.norm(x, dim=2) 
        
        loss = torch.tensor(0.0, device=x.device)
        
        
        mask_ring = (labels == 1)
        if mask_ring.any():
            r_ring = radii[mask_ring]
            loss += torch.mean((r_ring - 1.0) ** 2)
            
        mask_disk = (labels == 0)
        if mask_disk.any():
            r_disk = radii[mask_disk]
            loss += torch.mean(torch.relu(r_disk - 1.0) ** 2)

        return loss