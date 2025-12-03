import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import MLPDiffusion


TIMESTEPS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
FRAMES = 60 
betas = torch.linspace(0.0001, 0.02, TIMESTEPS).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def p_sample(model, x, t, t_index, label_embed):
    
    beta_t = betas[t_index]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t_index]
    sqrt_recip_alpha_t = torch.sqrt(1.0 / alphas[t_index])
    
    t_tensor = torch.full((x.shape[0],), t_index, device=device, dtype=torch.long)
    
    
    return None 

def interpolate_diffusion(model, alpha):
    """
    alpha: 0.0 (Disk) -> 1.0 (Ring)
    """
    
    label_0 = torch.tensor([0], device=device)
    label_1 = torch.tensor([1], device=device)
    
    emb_0 = model.class_emb(label_0) 
    emb_1 = model.class_emb(label_1) 
    
    mixed_emb = (1 - alpha) * emb_0 + alpha * emb_1
    
    img = torch.randn((1, 500, 2), device=device)
    
    for i in reversed(range(0, TIMESTEPS)):
        
        t_index = i
        x = img
        
        beta_t = betas[t_index]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alphas[t_index])
        
        t_tensor = torch.full((x.shape[0],), t_index, device=device, dtype=torch.long)
        
        
        
        
        time_emb = model.time_mlp(t_tensor)
        
        
        c_emb = mixed_emb 
        
        x_in = model.head(torch.cat([
            x, 
            time_emb[:, None, :].expand(-1, x.shape[1], -1), 
            c_emb[:, None, :].expand(-1, x.shape[1], -1)
        ], dim=-1))
        
        h = model.blocks(x_in)
        noise_pred = model.tail(h)

        model_mean = sqrt_recip_alpha_t * (
            x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t
        )
        
        if i > 0:
            noise = torch.randn_like(x)
            img = model_mean + torch.sqrt(posterior_variance[t_index]) * noise
        else:
            img = model_mean
            
    return img[0].detach().cpu().numpy()

if __name__ == "__main__":
    print("Model loaded...")
    model = MLPDiffusion(num_classes=2, hidden_size=256).to(device)
    model.load_state_dict(torch.load("diffusion_model_conditioned.pth"))
    model.eval()
    
    print("Interpolation video is being prepared...")
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title("Manifold Interpolation: Disk -> Ring")
    scatter = ax.scatter([], [], s=10, c='purple', alpha=0.6)
    
    def update(frame):
        alpha = frame / FRAMES 
        
        points = interpolate_diffusion(model, alpha)
        
        scatter.set_offsets(points)
        ax.set_title(f"Transition: %{int(alpha*100)} (Disk -> Ring)")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=100, blit=True)
    
    ani.save('manifold_surf.gif', writer='pillow', fps=15)
    print("GIF saved: manifold_surf.gif")
    plt.show()