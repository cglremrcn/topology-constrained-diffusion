import torch
import torch.nn as nn
import numpy as np
from model import MLPDiffusion
import matplotlib.pyplot as plt

TIME_STEPS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

betas = torch.linspace(0.0001,0.02,TIME_STEPS).to(device)

alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim = 0).to(device)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def p_sample(model,x,t,t_index,label):


    beta_t = betas[t_index]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t_index]
    sqrt_recip_alpha_t = torch.sqrt(1.0 / alphas[t_index])
    

    t_tensor = torch.full((x.shape[0],), t_index, device=device, dtype=torch.long)
    label_tensor = torch.full((x.shape[0],), label, device=device, dtype=torch.long)
    model_mean = sqrt_recip_alpha_t * (x - beta_t * model(x,t_tensor,label_tensor) / sqrt_one_minus_alpha_cumprod_t)


    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance[t_index]) * noise




def sample(model,shape,target_label):

    img = torch.randn(shape,device=device)

    for i in reversed(range(0,TIME_STEPS)):
        img = p_sample(model,img,i,i,target_label)

        if i % 20 == 0:
            print(f"Step {i}")

    return img




if __name__ == "__main__":
    
    model = MLPDiffusion().to(device)
    try:
        model.load_state_dict(torch.load("diffusion_model_conditioned.pth"))
        print("Trained model loaded!")
    except:
        print("Error: 'diffusion_model_conditioned.pth' not found. Run train.py first!")
        exit()
        
    model.eval()

    
    with torch.no_grad():
        data_disk = sample(model,shape=(1,1000,2),target_label=0)
        data_ring = sample(model,shape=(1,1000,2),target_label=1)

    
    disk_np = data_disk[0].cpu().numpy()
    ring_np = data_ring[0].cpu().numpy()
    
    plt.figure(figsize=(10,7))
    

    plt.subplot(1,2,1)
    plt.scatter(disk_np[:,0], disk_np[:,1], s=5, c='red', alpha=0.6)
    plt.title("Generated Disk")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    plt.subplot(1,2,2)
    plt.scatter(ring_np[:,0], ring_np[:,1], s=5, c='blue', alpha=0.6)
    plt.title("Generated Ring")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("Drawing completed!")