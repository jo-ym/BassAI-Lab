import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dir_path = 'stage1/downloads'
output_file = 'stage2/emotion_features.csv'


# Store analysis results
results = []

# Process each audio file
files = os.listdir(dir_path)
for index, filename in enumerate(files):
    if filename.endswith('.wav'):
        filepath = os.path.join(dir_path, filename)
        print(f'[Processing {filename} ({index + 1} / {len(files)})]')
        
        # Load audio file
        print('Loading audio...')
        y, sr = librosa.load(filepath)
        
        # Extract features
        print('Extracting features...')
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo.item()
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr)).item()
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)).item()
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)).item()
        rms = np.mean(librosa.feature.rms(y=y)).item()
        
        # Compute emotional features
        valence = chroma * 10  # Simulated valence (simple model)
        arousal = tempo / 200  # Simulated arousal
        
        # Classify labels
        if valence > 6 and arousal > 0.6:
            label = 'Chill'
        elif valence < 4 and arousal < 0.4:
            label = 'Dark'
        elif 4 <= valence <= 6 and 0.4 <= arousal <= 0.6:
            label = 'Neutral'
        else:
            label = 'Relax'
        
        # Store results
        results.append([filename, tempo, chroma, zcr, spectral_centroid, rms, valence, arousal, label])


print("Analysis completed. Saving results...")

# Convert results to DataFrame
df = pd.DataFrame(results, columns=['File', 'Tempo', 'Chroma', 'ZCR', 'Centroid', 'RMS', 'Valence', 'Arousal', 'Label'])
# Ensure columns are correct types
df['Tempo'] = df['Tempo'].astype(float)
df['Chroma'] = df['Chroma'].astype(float)
df['ZCR'] = df['ZCR'].astype(float)
df['Centroid'] = df['Centroid'].astype(float)
df['RMS'] = df['RMS'].astype(float)
df['Valence'] = df['Valence'].astype(float)
df['Arousal'] = df['Arousal'].astype(float)
df['Label'] = df['Label'].astype(str)
df.to_csv(output_file, index=False)
print(f'Results saved to {output_file}')

# Plot tag distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Label')
plt.title('Tag Distribution')
plt.xlabel('Tags')
plt.ylabel('Count')
plt.savefig('stage2/tag_distribution.png')
plt.show()

# Plot emotional features
# Ensure Label column format
df['Label'] = df['Label'].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
df['Label'] = pd.Categorical(df['Label'])
plt.figure(figsize=(6, 6))
if len(df['Label'].unique()) == 1:
    sns.scatterplot(data=df, x='Valence', y='Arousal', color='blue')
else:
    sns.scatterplot(data=df, x='Valence', y='Arousal', hue='Label', style='Label')
plt.title('Valence vs Arousal')
plt.xlabel('Valence (Pleasantness)')
plt.ylabel('Arousal (Energy)')
plt.grid(True)
plt.savefig('stage2/valence_arousal_scatter.png')
plt.show()