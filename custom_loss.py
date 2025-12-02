import torch
import torch.nn as nn
from torch_topological.nn import VietorisRipsComplex

class TopologicalLoss(nn.Module):
    def __init__(self,dim = 1):
        super().__init__()

        self.vr = VietorisRipsComplex(dim = dim)

    def forward(self,x,labels):

        diagrams = self.vr(x)

        loss = torch.tensor(0.0,device = x.device)

        count = 0

        for i,diagram in enumerate(diagrams):

            if diagram is None or len(diagram) == 0:
                continue

            

            lifetimes = diagram[:,1] - diagram[:,0]


            target = labels[i]

            if target == 1:

                if len(lifetimes) == 0:
                    max_lifetime = torch.max(lifetimes)

                    loss -= max_lifetime

                    count += 1

                else:

                    loss += torch.sum(lifetimes**2)
                    count += 1

        
        if count > 0:
            return loss / count
        else:
            return loss
