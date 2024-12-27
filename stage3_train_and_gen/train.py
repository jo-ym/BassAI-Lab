import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet2DModel
from sklearn.manifold import TSNE
import time


# Training Parameters
DATA_DIR = 'stage1/downloads'
BATCH_SIZE = 64
EPOCHS = 10
LATENT_DIM = 64
LEARNING_RATE = 1e-4
SAMPLE_RATE = 16000
N_MELS = 64
MAX_LEN = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)


# Dataset Processing
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max(0, MAX_LEN - mel_spec.shape[1]))), mode='constant')
        mel_spec = mel_spec[:, :MAX_LEN]
        return torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        z_params = self.encoder(x)
        mu, log_var = z_params[:, :LATENT_DIM], z_params[:, LATENT_DIM:]
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

vae = VAE(N_MELS * MAX_LEN, LATENT_DIM).to(DEVICE)
optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

# Diffusion Model
model = UNet2DModel(
    sample_size=(N_MELS, MAX_LEN),
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(DEVICE)
optimizer_diff = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Process
dataset = AudioDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Starting training...")
start_time = time.time()
for epoch in range(EPOCHS):
    vae.train()
    model.train()
    epoch_loss_vae = 0
    epoch_loss_diff = 0
    
    for batch_idx, data in enumerate(dataloader):
        batch_start_time = time.time()
        # VAE Training
        data_flat = data.view(data.size(0), -1)
        reconstructed, mu, log_var = vae(data_flat)
        recon_loss = loss_function(reconstructed, data_flat)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss_vae = recon_loss + kl_loss
        optimizer_vae.zero_grad()
        loss_vae.backward()
        optimizer_vae.step()
        epoch_loss_vae += loss_vae.item()
        
        # Diffusion Training
        noise = torch.randn_like(data)
        noisy_data = data + noise * 0.1
        timestep = torch.randint(0, 1000, (data.size(0),), device=DEVICE)  # Added timestep
        pred = model(noisy_data, timestep).sample
        loss_diff = loss_function(pred, data)
        optimizer_diff.zero_grad()
        loss_diff.backward()
        optimizer_diff.step()
        epoch_loss_diff += loss_diff.item()
        
        batch_end_time = time.time()
        print(f"Epoch [{epoch+1}/{EPOCHS}], VAE Loss: {epoch_loss_vae / len(dataloader):.4f}, Diff Loss: {epoch_loss_diff / len(dataloader):.4f}, Batch [{batch_idx+1}/{len(dataloader)}], Time: {batch_end_time - batch_start_time:.2f}s")
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], VAE Loss: {epoch_loss_vae / len(dataloader):.4f}, Diff Loss: {epoch_loss_diff / len(dataloader):.4f}")

end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f}s")


# Latent Space Visualization
vae.eval()
with torch.no_grad():
    z = torch.randn(100, LATENT_DIM).cpu().detach().numpy()
    z_embedded = TSNE(n_components=2).fit_transform(z)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c='blue', alpha=0.5)
    plt.title("Latent Space Visualization (TSNE)")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('stage3/latent_space_visualization.png')
    plt.close()

# Save Models
torch.save(vae.state_dict(), 'stage3/vae_model.pth')
torch.save(model.state_dict(), 'stage3/diff_model.pth')
print("Models saved!")
