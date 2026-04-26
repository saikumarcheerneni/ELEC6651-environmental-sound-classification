import numpy as np
import librosa

def load_audio(path, sr=22050, mono=True, duration=5.0):
    y, _ = librosa.load(path, sr=sr, mono=mono)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr

def stats_aggregate(feature_2d):
    mean = np.mean(feature_2d, axis=1)
    std = np.std(feature_2d, axis=1)
    return np.concatenate([mean, std], axis=0)

def extract_feature_vector(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    rms = librosa.feature.rms(S=S)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

    parts = [
        stats_aggregate(mfcc),
        stats_aggregate(centroid),
        stats_aggregate(bandwidth),
        stats_aggregate(rolloff),
        stats_aggregate(rms),
        stats_aggregate(zcr),
    ]
    return np.concatenate(parts, axis=0)

def feature_names(n_mfcc=13):
    names = []
    for i in range(n_mfcc):
        names.append(f"mfcc_{i+1}_mean")
    for i in range(n_mfcc):
        names.append(f"mfcc_{i+1}_std")
    for base in ["centroid", "bandwidth", "rolloff", "rms", "zcr"]:
        names += [f"{base}_mean", f"{base}_std"]
    return names