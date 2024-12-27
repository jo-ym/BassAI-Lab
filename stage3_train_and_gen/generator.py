import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from diffusers import UNet2DModel
from sklearn.manifold import TSNE


# Parameters
LATENT_DIM = 64
SAMPLE_RATE = 16000
N_MELS = 64
MAX_LEN = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)


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

# Load Models
vae = VAE(N_MELS * MAX_LEN, LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load('stage3/vae_model.pth'))
vae.eval()



model = UNet2DModel(
    sample_size=(N_MELS, MAX_LEN),
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(DEVICE)
model.load_state_dict(torch.load('stage3/diff_model.pth'))
model.eval()

# Generate Data
with torch.no_grad():
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    generated_flat = vae.decoder(z)
    
    generated_flat = generated_flat.view(1, 1, N_MELS, MAX_LEN)
    
    timestep = torch.randint(0, 1000, (1,), device=DEVICE)
    noise = torch.randn_like(generated_flat).to(DEVICE)
    generated_output = model(noise, timestep).sample

# Save Generated Results
np.save('stage3/generated_mel.npy', generated_output.cpu().squeeze().numpy())
print("Mel spectrogram saved!")

# Convert to Audio
def griffin_lim_reconstruction(mel_spec):
    mel_spec = librosa.db_to_power(mel_spec)
    waveform = librosa.feature.inverse.mel_to_audio(
        mel_spec, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, win_length=1024
    )
    return waveform

mel_spec = generated_output.cpu().squeeze().numpy()
waveform = griffin_lim_reconstruction(mel_spec)

sf.write('stage3/generated_audio.wav', waveform, SAMPLE_RATE)


# TSNE Visualization
z = torch.randn(100, LATENT_DIM).to(DEVICE)
z_embedded = TSNE(n_components=2, perplexity=30).fit_transform(z.cpu().numpy())
plt.figure(figsize=(8, 6))
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c='red', alpha=0.5)
plt.title("Generated Latent Space Visualization (TSNE)")
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('stage3/generated_latent_space_visualization.png')
