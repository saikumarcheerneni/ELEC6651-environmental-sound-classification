"""
generate_spectrogram.py
=======================
Run this from your project root to generate a spectrogram figure for your report.

Usage:
    python generate_spectrogram.py --audio data/ESC-50/audio/1-100032-A-0.wav

It will save:  results/spectrogram_example.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def generate(audio_path, out_path="results/spectrogram_example.png"):
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=5.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle("Time-Frequency Analysis using STFT", fontsize=14, fontweight="bold", y=0.98)

    # 1. Waveform
    ax1 = axes[0]
    times = np.linspace(0, len(y)/sr, len(y))
    ax1.plot(times, y, color="#534AB7", linewidth=0.6, alpha=0.85)
    ax1.fill_between(times, y, alpha=0.3, color="#534AB7")
    ax1.set_title("(a) Raw Waveform  x[n]", fontsize=11)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, len(y)/sr)
    ax1.spines[["top","right"]].set_visible(False)

    # 2. STFT Spectrogram
    ax2 = axes[1]
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    img2 = librosa.display.specshow(D, sr=sr, hop_length=512,
                                    x_axis="time", y_axis="hz",
                                    ax=ax2, cmap="magma")
    fig.colorbar(img2, ax=ax2, format="%+2.0f dB", pad=0.01)
    ax2.set_title("(b) STFT Spectrogram  — time-frequency representation", fontsize=11)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    # 3. Mel-frequency spectrogram
    ax3 = axes[2]
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                        n_fft=2048, hop_length=512)
    M_db = librosa.power_to_db(M, ref=np.max)
    img3 = librosa.display.specshow(M_db, sr=sr, hop_length=512,
                                    x_axis="time", y_axis="mel",
                                    ax=ax3, cmap="viridis")
    fig.colorbar(img3, ax=ax3, format="%+2.0f dB", pad=0.01)
    ax3.set_title("(c) Mel-frequency Spectrogram  — perceptual frequency scale", fontsize=11)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Mel Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved spectrogram to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to any ESC-50 .wav file")
    parser.add_argument("--out", default="results/spectrogram_example.png")
    args = parser.parse_args()
    generate(args.audio, args.out)