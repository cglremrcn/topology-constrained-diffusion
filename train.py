import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
from model import MLPDiffusion
import matplotlib.pyplot as plt

# Hyperparameters
DATASET_SIZE = 10000
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 50
TIME_STEPS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



betas = torch.linspace(0.0001,0.02,TIME_STEPS).to(device)

alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim = 0)


sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)


def q_sample(x_0,t,noise=None):

    if noise is None:
        noise = torch.randn_like(x_0)
    

    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1,1,1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1,1,1)
    
    return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise ,noise



dataset = PointCloudDataset(DATASET_SIZE)
data_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

model = MLPDiffusion().to(device)

optimizer = optim.Adam(model.parameters(),lr = LR)
criterion = nn.MSELoss()


loss_history = []

for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_data, _ in data_loader:

        x_0 = batch_data.to(device)

        t = torch.randint(0,TIME_STEPS,(x_0.shape[0],),device=device)   

        noise = torch.randn_like(x_0)

        x_t, noise_added = q_sample(x_0,t,noise)        
        

        noise_pred = model(x_t,t).to(device)

        loss = criterion(noise_pred,noise_added)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(data_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0:

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss}")
        

torch.save(model.state_dict(),"diffusion_model.pth")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

