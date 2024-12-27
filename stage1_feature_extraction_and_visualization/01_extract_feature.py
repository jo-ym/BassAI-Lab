import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import json
from turbolpc import analysis


def extract_formants(y, sr, order=12):
    """
    Extract Formant features using TurboLPC.
    :param y: Audio signal
    :param sr: Sampling rate
    :param order: LPC order
    :return: Extracted Formant frequencies
    """
    lpc_coeffs = analysis.arburg_vector(x=y, order=order)
    
    # Calculate LPC roots, whose real parts correspond to Formant frequencies
    roots = np.roots(lpc_coeffs[0])
    positive_roots = [r for r in roots if np.imag(r) >= 0]  # Select roots with positive real parts
    angles = np.angle(positive_roots)
    formants = sorted([angle * (sr / (2 * np.pi)) for angle in angles])
    
    return formants

def extract_features(filepath):
    print(f"[Analysis] Analyzing file: {filepath}")
    # Load the audio and normalize the amplitude
    y, sr = librosa.load(filepath, sr=44100)  # Sampling rate 44.1kHz
    y = librosa.util.normalize(y)  # Amplitude normalization [-1, 1]
    
    # LPC Formant analysis
    print("[Progress] Analyzing LPC Formant ...")
    formants = extract_formants(y, sr)
    
    # Onset detection
    print("[Progress] Onset Detection")
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    
    # Harmonic/Percussive separation
    print("[Progress] Harmonic/Percussive Separation")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Frequency and timbre features
    print("[Progress] Extracting frequency and timbre features...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = StandardScaler().fit_transform(mfcc.T).T
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = MinMaxScaler(feature_range=(0, 1)).fit_transform(chroma)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid = StandardScaler().fit_transform(spectral_centroid.T).T
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast = StandardScaler().fit_transform(spectral_contrast.T).T
    
    # Rhythm and dynamics features
    print("[Progress] Analyzing rhythm and dynamics...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats_time = librosa.frames_to_time(beats, sr=sr).tolist()  # Convert beats to timestamps
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    zero_crossings = MinMaxScaler(feature_range=(0, 1)).fit_transform(zero_crossings)
    
    # Energy and spectral density features
    print("[Progress] Calculating energy and spectral density...")
    rms = librosa.feature.rms(y=y)
    rms = StandardScaler().fit_transform(rms.T).T
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec = MinMaxScaler(feature_range=(0, 1)).fit_transform(mel_spec)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth = StandardScaler().fit_transform(spectral_bandwidth.T).T
    
    # Pitch and fundamental frequency analysis
    print("[Progress] Analyzing pitch and fundamental frequency...")
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    magnitude_mean = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0 
    
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-6)
    
    print(f"[Completed] Analysis finished for: {filepath}")
    
    # Output all features
    return {
        "filename": os.path.basename(filepath),
        "onset_times": onset_times.tolist(),
        "y_harmonic": y_harmonic.tolist(),
        "y_percussive": y_percussive.tolist(),
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
        "chroma_mean": np.mean(chroma, axis=1).tolist(),
        "spectral_centroid_mean": np.mean(spectral_centroid).tolist(),
        "spectral_contrast_mean": np.mean(spectral_contrast).tolist(),
        "tempo": float(tempo.item()) if np.isscalar(tempo) == False else float(tempo),
        "beats_time": beats_time,
        "zero_crossings_mean": np.mean(zero_crossings).tolist(),
        "rms_mean": np.mean(rms).tolist(),
        "mel_spec_mean": np.mean(mel_spec).tolist(),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth).tolist(),
        "pitch_mean": float(pitch_mean),
        "magnitude_mean": float(magnitude_mean),
        "hnr": float(hnr),
        "formants": formants 
    }

# Batch process all audio files in a directory
def process_directory(input_dir, output_file="audio_features.json"):
    features_list = []
    
    # Process all .wav files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):  # Process only WAV files
            filepath = os.path.join(input_dir, filename)
            print(f"Processing file: {filename}")
            try:
                # Execute feature extraction function
                features = extract_features(filepath)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Save features to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(features_list, f, ensure_ascii=False, indent=4)
    
    print(f"All audio features have been saved to {output_file}")


def process_directory_separate(input_dir, output_dir="features"):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    
    # Process all .wav files
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing file: {filename}")
            try:
                # Extract features
                features = extract_features(filepath)
                
                # Save to a separate JSON file
                output_file = os.path.join(output_dir, f"{filename}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(features, f, ensure_ascii=False, indent=4)
                
                print(f"Features saved: {output_file}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


# process_directory('stage1/downloads')
process_directory_separate(input_dir="stage1/downloads", output_dir="stage1/features")
