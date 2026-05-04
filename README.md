# Environmental Sound Classification
### Using Statistical Features and Machine Learning

**ELEC 6651 — Advanced Signal Processing | Concordia University**  
**Submitted by:** Saikumar Cheerneni

---

## Project Overview

This project builds a complete **environmental sound classification system** that listens to a 5-second audio clip and identifies which of **50 possible sound classes** it belongs to — such as dog bark, rain, siren, clapping, or thunderstorm.

The system uses **classical signal processing** for feature extraction combined with **three machine learning classifiers**, achieving a peak accuracy of **62.4%** — competitive with the published benchmark of 64.5% (Piczak, 2015).

---

## Results Summary

| Classifier | Best Parameters | Accuracy | vs. Random Baseline (2%) |
|---|---|---|---|
| k-NN | k = 5 | 45.4% | 22.7× |
| Random Forest | 200 trees, depth = None | 55.9% | 27.9× |
| **SVM (RBF)** | **C=10, γ=scale** | **62.4% ★** | **31.2×** |

> Piczak (2015) published benchmark using classical ML: **64.5%**

---

## System Pipeline

```
Raw Audio (.wav)
      ↓
Step 1 — Audio loading & resampling
         (mono · 22,050 Hz · 5 seconds · 110,250 samples)
      ↓
Step 2 — STFT (Short-Time Fourier Transform)
         (N=2,048 · H=512 hop · ~216 frames · 1,025 freq bins)
      ↓
Step 3 — Feature extraction
         (MFCCs · centroid · bandwidth · rolloff · RMS · ZCR)
      ↓
Step 4 — StandardScaler normalisation
         (zero mean · unit variance)
      ↓
36-dimensional feature vector per clip
         (26 MFCCs + 10 other = 36 total)
      ↓
Three classifiers trained in parallel
         (k-NN · SVM-RBF · Random Forest)
      ↓
5-fold stratified cross-validation
         (1,600 train · 400 test per fold)
      ↓
Predicted sound class (1 of 50)
```

---

## Features Extracted

| Feature | What it measures | Dimensions |
|---|---|---|
| MFCCs (n=13) | Spectral envelope shape on mel scale | 26 |
| Spectral Centroid | Brightness — high for sirens, low for thunder | 2 |
| Spectral Bandwidth | Spread — narrow for tonal, wide for rain/noise | 2 |
| Spectral Rolloff | Frequency below which 85% of energy sits | 2 |
| RMS Energy | Loudness — high spike for bark, stable for rain | 2 |
| Zero-Crossing Rate | Sign changes — low for tonal, high for noisy | 2 |
| **Total** | | **36** |

Each feature is aggregated as **mean + standard deviation** across all ~216 STFT frames.

---

## Project Structure

```
env_sound_cls_project/
├── src/
│   ├── utils.py                  # Audio loading + feature extraction functions
│   ├── extract_features.py       # Batch feature extraction → results/features.csv
│   └── train_models.py           # Model training, CV evaluation, confusion matrices
├── results/
│   ├── features.csv              # Extracted 36-D feature matrix (generated)
│   ├── metrics.json              # Accuracy + best params for all 3 models
│   ├── confusion_svm_rbf.png     # Confusion matrix — SVM (best model)
│   ├── confusion_knn.png         # Confusion matrix — k-NN
│   ├── confusion_random_forest.png  # Confusion matrix — Random Forest
│   ├── report_svm_rbf.csv/txt    # Per-class classification report — SVM
│   ├── report_knn.csv/txt        # Per-class classification report — k-NN
│   └── report_random_forest.csv/txt # Per-class classification report — RF
├── audio_dashboard.py            # Real-time Gradio web dashboard
├── generate_spectrogram.py       # Spectrogram visualisation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv .env
.env\Scripts\activate

# Mac / Linux
python -m venv .env
source .env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the ESC-50 dataset

Download from: https://github.com/karolpiczak/ESC-50

Place it so the folder structure looks like this:

```
env_sound_cls_project/
└── data/
    └── ESC-50/
        ├── audio/        ← 2,000 .wav files go here
        └── meta/
            └── esc50.csv ← metadata file
```

---

## How to Run

### Step 1 — Extract features

```bash
python src/extract_features.py --esc_root data/ESC-50
```

This reads all 2,000 audio clips, extracts the 36-dimensional feature vector for each, and saves the result to `results/features.csv`.

**Output:** `results/features.csv` — a table of 2,000 rows × 36 feature columns + label + fold

---

### Step 2 — Train and evaluate all models

```bash
python src/train_models.py
```

This trains k-NN, SVM-RBF, and Random Forest using 5-fold stratified cross-validation with GridSearchCV. Results, confusion matrices, and classification reports are saved to the `results/` folder.

**Outputs:**
- `results/metrics.json` — accuracy and best hyperparameters for each model
- `results/confusion_*.png` — confusion matrix images
- `results/report_*.csv/txt` — per-class precision, recall, F1 scores

---

### Step 3 — Launch the live dashboard (optional)

```bash
python audio_dashboard.py
```

Opens a Gradio web app at **http://localhost:7860**

Upload any audio file and all three trained classifiers will predict the sound class instantly. The dashboard also shows:
- Audio waveform
- STFT spectrogram
- Feature radar chart
- Top-5 predictions per model
- All 36 raw feature values

---

### Step 4 — Generate spectrogram visualisation (optional)

```bash
python generate_spectrogram.py
```

Saves `results/spectrogram_example.png` showing the raw waveform, STFT spectrogram, and mel-frequency spectrogram for an example clip.

---

## Key Technical Details

### Why STFT instead of plain FFT?
Environmental sounds are **nonstationary** — their frequency content changes over time. A single global FFT on the full 5-second clip collapses all time information into one blurred average. STFT cuts the signal into 93ms windows and transforms each one separately, producing a 2D time-frequency map (spectrogram).

### Why mean + standard deviation?
Each feature produces 216 values (one per STFT frame). To create a **fixed-length** vector for classification, we compute the mean and standard deviation across all frames. The mean captures the average character of the sound; the std captures how much it varies over time.

### Fundamental limitation
Mean and std are **time-invariant** — a dog bark played backwards produces the identical 36 numbers. Temporal order is completely lost. This explains why acoustically similar pairs (dog vs cat, rain vs water_drops) confuse the classifier.

### Why SVM wins
The RBF kernel implicitly maps the 36-D features to an infinite-dimensional space where the 50 classes become linearly separable. The maximum-margin principle also reduces overfitting on the small (2,000 sample) dataset.

---

## Confusion Matrix Highlights

| Pair | Why confused |
|---|---|
| dog ↔ cat | Both short animal vocalisations with overlapping MFCC profiles |
| rain ↔ water_drops | Both broadband noise with nearly identical ZCR and flat spectral profile |
| breathing ↔ snoring | Both low-frequency periodic respiratory sounds |

**Easy classes:** siren, clock_tick, thunderstorm — each has a unique spectral fingerprint that no other class shares.

---

## Future Improvements

1. **Delta MFCCs** — add first and second derivatives of MFCC coefficients to capture temporal dynamics (expected +5–8% accuracy)
2. **More MFCCs** — increase from 13 to 40 coefficients for finer spectral resolution
3. **CNN on mel-spectrograms** — treat the spectrogram as an image; CNNs achieve ~90% on ESC-50

---

## Dependencies

```
numpy
pandas
librosa
soundfile
scikit-learn
matplotlib
tqdm
gradio
```

Install all with: `pip install -r requirements.txt`

---

## References

1. K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," *Proc. 23rd ACM International Conference on Multimedia*, 2015. https://github.com/karolpiczak/ESC-50
2. A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, 2011. https://scikit-learn.org
4. B. McFee et al., "librosa: Audio and Music Signal Analysis in Python," *Proc. 14th Python in Science Conference*, 2015. https://librosa.org

---

## License

This project was developed for academic purposes as part of ELEC 6651 — Advanced Signal Processing at Concordia University, Montreal, Canada.