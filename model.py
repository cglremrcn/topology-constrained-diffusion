import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device  = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.ff = nn.Linear(size,size)
        self.act = nn.GELU()


    def forward(self,x):
        return self.act(self.ff(x)) 



class MLPDiffusion(nn.Module):
    def __init__(self,hidden_size = 128,num_classes = 2):
        super().__init__()


        self.head = nn.Linear(2,hidden_size)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_size),
            nn.Linear(hidden_size,hidden_size),
            nn.GELU()
        )
        
        self.class_emb = nn.Embedding(num_classes,hidden_size)

        self.blocks = nn.Sequential(
            Block(hidden_size),
            Block(hidden_size),
            Block(hidden_size)
        )

        self.tail = nn.Linear(hidden_size,2)

    def forward(self,x,t,labels):
        
        time_emb = self.time_mlp(t)

        c_emb = self.class_emb(labels)

        x = self.head(x)

        x = x + time_emb[:,None,:] + c_emb[:,None,:]

        x = self.blocks(x)

        x = self.tail(x)
        return x



if __name__ == "__main__":

    model = MLPDiffusion(hidden_size=128)

    dummy_x = torch.randn(10, 2)

    dummy_t = torch.randint(0, 100, (10,))
    
    dummy_labels = torch.randint(0, 2, (10,))

    output = model(dummy_x, dummy_t, dummy_labels)
    

    print(f"Input Shape: {dummy_x.shape}") 
    print(f"Output Shape: {output.shape}") 
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params}")

    print("model is updated")