import torch
import matplotlib.pyplot as plt
import gudhi
import numpy as np
from model import MLPDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 100

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

def generate_one(model, label):
    img = torch.randn((1, 500, 2), device=device)
    for i in reversed(range(0, TIMESTEPS)):
        img = p_sample(model, img, i, i, label)
    return img[0].cpu().numpy()

def get_betti(points):
    rips = gudhi.RipsComplex(points=points, max_edge_length=0.5)
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()
    betti = st.betti_numbers()
    if len(betti) == 1: betti.append(0) 
    return betti[:2]

if __name__ == "__main__":
    model = MLPDiffusion(num_classes=2).to(device)
    model.load_state_dict(torch.load("diffusion_model_conditioned.pth"))
    model.eval()

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Models Generated Rings and Betti Numbers", fontsize=16)

    print("Generating...")
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                points = generate_one(model, label=1)
                betti = get_betti(points)
                
                ax = axs[i, j]
                ax.scatter(points[:,0], points[:,1], s=2, c='blue')
                
                title_color = 'green' if betti == [1, 1] else 'red'
                ax.set_title(f"Betti: {betti}", color=title_color, fontweight='bold')
                ax.axis('equal')
                ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    print("Done!")