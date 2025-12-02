import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gudhi
from scipy.spatial.distance import pdist, squareform
from model import MLPDiffusion


TIMESTEPS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    if t_index == 0: return model_mean
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance[t_index]) * noise

def generate_from_model(label):
    
    print("Model is loading")
    model = MLPDiffusion(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load("diffusion_model_conditioned.pth"))
    except:
        print("Model file is not found! First run train.py.")
        exit()
    model.eval()
    

    print(f"{'Ring' if label==1 else 'Circle'} is requested from model")
    with torch.no_grad():
        img = torch.randn((1, 100, 2), device=device) 
        for i in reversed(range(0, TIMESTEPS)):
            img = p_sample(model, img, i, i, label)
            
    return img[0].cpu().numpy()



points = generate_from_model(label=1)


print("calculating the model output topology...")
rips = gudhi.RipsComplex(points=points)
st = rips.create_simplex_tree(max_dimension=2)
persistence = st.persistence()


fig, ax = plt.subplots(figsize=(9,9))
ax.set_title("Topology Test")
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

scatter = ax.scatter(points[:, 0], points[:, 1], s=50, c='blue', zorder=10, label="Model Output")
lines = [] 
dist_matrix = squareform(pdist(points))

status_text = ax.text(0, 1.2, "", ha='center', fontsize=12, fontweight='bold', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

def update(frame):
    epsilon = frame * 0.03 
    
    
    h0_count = 0; h1_count = 0
    for interval in persistence:
        dim = interval[0]; birth = interval[1][0]; death = interval[1][1]
        if birth <= epsilon and death > epsilon:
            if dim == 0: h0_count += 1
            if dim == 1: h1_count += 1
            
    status_text.set_text(f"Radius: {epsilon:.2f}\n"
                         f"H0 (Component): {h0_count} | H1 (Cycle): {h1_count}")
    
    if h1_count == 1: 
        status_text.set_bbox(dict(facecolor='#ccffcc', edgecolor='green'))
    elif h1_count > 1: 
        status_text.set_bbox(dict(facecolor='#ffcccc', edgecolor='red'))
    else:
        status_text.set_bbox(dict(facecolor='white', edgecolor='gray'))


    scatter.set_sizes([50 + (epsilon * 60)])
    for line in lines: line.remove()
    lines.clear()
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dist_matrix[i, j] < epsilon * 2:
                line, = ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 
                                c='red', alpha=0.3, lw=1)
                lines.append(line)

print("start the animation...")
# Slow down animation: interval=200ms (5 fps)
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 60), interval=200, repeat=False)

# Set ffmpeg path explicitly since it's not in system PATH
plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\BlueStacks_nxt\ffmpeg.exe'

# Save as MP4 video
print("Saving animation as MP4 video...")
try:
    # fps=5 matches the interval of 200ms
    ani.save('topology_filtration.mp4', writer='ffmpeg', fps=5, dpi=100)
    print("Video saved as 'topology_filtration.mp4'")
except Exception as e:
    print(f"Could not save MP4: {e}")
    # Fallback to GIF if MP4 fails
    print("Saving animation as GIF...")
    try:
        ani.save('topology_filtration.gif', writer='pillow', fps=5)
        print("GIF saved as 'topology_filtration.gif'")
    except Exception as ex:
        print(f"Could not save GIF: {ex}")

# plt.show() # Commented out to run in background

print("Done.")