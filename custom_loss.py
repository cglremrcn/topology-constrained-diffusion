import torch
import torch.nn as nn
from torch_topological.nn import VietorisRipsComplex

class TopologicalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vr = VietorisRipsComplex(dim=1)

    def forward(self, x, labels):
        batch_results = self.vr(x)
        
        loss = torch.tensor(0.0, device=x.device)
        count = 0

        for i, dim_list in enumerate(batch_results):
            diagram = None
            
            for info in dim_list:
                if info.dimension == 1:
                    diagram = info.diagram
                    break
            
            if diagram is None or len(diagram) == 0:
                continue
            
            if not isinstance(diagram, torch.Tensor):
                diagram = torch.tensor(diagram, device=x.device, dtype=torch.float32)

            if len(diagram) == 0: continue

           
            lifetimes = diagram[:, 1] - diagram[:, 0]
            
            target = labels[i]
            
            if target == 1: 
                if len(lifetimes) > 0:
                    loss -= torch.max(lifetimes) 
                    count += 1
            
            else: 
                loss += torch.sum(lifetimes ** 2)
                count += 1
                
        if count > 0:
            return loss / count
        else:
            return loss