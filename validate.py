import torch
import numpy as np
import gudhi
from model import MLPDiffusion

TIMESTEPS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


SAMPLE_COUNT = 100 
betas = torch.linspace(0.0001, 0.02, TIMESTEPS).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def p_sample(model, x, t, t_index, label):

    beta_t = betas[t_index]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t_index]
    sqrt_recip_alpha_t = torch.sqrt(1.0 / alphas[t_index])
    
    t_tensor = torch.full((x.shape[0],), t_index, device=device, dtype=torch.long)
    label_tensor = torch.full((x.shape[0],), label, device=device, dtype=torch.long)
    
    model_mean = sqrt_recip_alpha_t * (
        x - beta_t * model(x, t_tensor, label_tensor) / sqrt_one_minus_alpha_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance[t_index]) * noise

def generate_shape(model, label):
    
    img = torch.randn((1, 500, 2), device=device) 
    for i in reversed(range(0, TIMESTEPS)):
        img = p_sample(model, img, i, i, label)
    return img[0].cpu().numpy()

def check_betti(point_cloud):

    rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=0.5)
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()
    betti = st.betti_numbers()

    if len(betti) == 1: betti.append(0)
    return betti[:2] 

if __name__ == "__main__":
    print("Model loading...")
    model = MLPDiffusion(num_classes=2).to(device)
    model.load_state_dict(torch.load("diffusion_model_conditioned.pth"))
    model.eval()

    print(f"\n--- Testing ({SAMPLE_COUNT} samples) ---")
    

    correct_rings = 0
    print("ring test...")
    
    with torch.no_grad():
        for i in range(SAMPLE_COUNT):
            points = generate_shape(model, label=1)
            betti = check_betti(points)
            
            if betti == [1, 1]:
                correct_rings += 1

            if (i+1) % 20 == 0: print(f"{i+1} samples completed...")


    correct_disks = 0
    print("\ndisk test...")
    
    with torch.no_grad():
        for i in range(SAMPLE_COUNT):
            points = generate_shape(model, label=0)
            betti = check_betti(points)
            
            if betti == [1, 0]:
                correct_disks += 1
            elif betti == [1]: 
                correct_disks += 1

            if (i+1) % 20 == 0: print(f"{i+1} samples completed...")


    print("\n" + "="*30)
    print(f"TEST REPORT")
    print("="*30)
    print(f"Ring Success: %{correct_rings} (Target: [1, 1])")
    print(f"Disk Success:  %{correct_disks}  (Target: [1, 0])")
    print("="*30)