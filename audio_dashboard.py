"""
Environmental Sound Classification Dashboard
============================================
Drop any .wav or .mp3 file → get instant predictions from your trained models.

Run:  python audio_dashboard.py
Then open:  http://localhost:7860
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import librosa.display
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ── Try importing Gradio ──────────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    print("Installing gradio …")
    os.system("pip install gradio --quiet")
    import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FEATURE EXTRACTION  (same logic as your utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path, sr=22050, duration=5.0):
    y, _ = librosa.load(path, sr=sr, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y, sr


def stats(feat_2d):
    return np.concatenate([np.mean(feat_2d, axis=1),
                           np.std(feat_2d,  axis=1)])


def extract_features(y, sr, n_mfcc=13, n_fft=2048, hop=512):
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    S         = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    centroid  = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    rms       = librosa.feature.rms(S=S)
    zcr       = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    return np.concatenate([stats(mfcc), stats(centroid), stats(bandwidth),
                           stats(rolloff), stats(rms), stats(zcr)])


def feature_names(n_mfcc=13):
    names = [f"mfcc_{i+1}_mean" for i in range(n_mfcc)]
    names += [f"mfcc_{i+1}_std"  for i in range(n_mfcc)]
    for b in ["centroid", "bandwidth", "rolloff", "rms", "zcr"]:
        names += [f"{b}_mean", f"{b}_std"]
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL LOADING / TRAINING
# ─────────────────────────────────────────────────────────────────────────────

MODELS_FILE  = "results/trained_models.pkl"
FEATURES_CSV = "results/features.csv"
METRICS_JSON = "results/metrics.json"

_models  = {}          # populated by load_or_train()
_classes = []

def load_or_train():
    global _models, _classes

    # ── A: load pre-trained pickled models ──
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, "rb") as f:
            data = pickle.load(f)
        _models  = data["models"]
        _classes = data["classes"]
        print(f"[dashboard] Loaded {len(_models)} models from {MODELS_FILE}")
        return "loaded"

    # ── B: re-train from features.csv ──
    if os.path.exists(FEATURES_CSV):
        print("[dashboard] Training models from features.csv …")
        df = pd.read_csv(FEATURES_CSV)
        y_all  = df["label"].values
        _classes = sorted(df["label"].unique().tolist())
        feat_cols = [c for c in df.columns if c not in ["label","fold","filename"]]
        X_all = df[feat_cols].values

        specs = {
            "k-NN":           Pipeline([("sc", StandardScaler()),
                                        ("clf", KNeighborsClassifier(n_neighbors=5))]),
            "SVM (RBF)":      Pipeline([("sc", StandardScaler()),
                                        ("clf", SVC(kernel="rbf", C=10,
                                                    gamma="scale", probability=True))]),
            "Random Forest":  Pipeline([("clf", RandomForestClassifier(
                                                    n_estimators=200, random_state=42))]),
        }
        for name, pipe in specs.items():
            pipe.fit(X_all, y_all)
            _models[name] = pipe
            print(f"  ✓ {name}")

        os.makedirs("results", exist_ok=True)
        with open(MODELS_FILE, "wb") as f:
            pickle.dump({"models": _models, "classes": _classes}, f)
        return "trained"

    return "no_data"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PURPLE  = "#534AB7"
TEAL    = "#1D9E75"
AMBER   = "#BA7517"
CORAL   = "#D85A30"
GRAY_BG = "#F8F7FF"
DARK    = "#1C1B2E"

MODEL_COLORS = {
    "k-NN":          TEAL,
    "SVM (RBF)":     PURPLE,
    "Random Forest": AMBER,
}

def make_waveform_fig(y, sr):
    fig, ax = plt.subplots(figsize=(7, 1.8), facecolor=GRAY_BG)
    ax.set_facecolor(GRAY_BG)
    times = np.linspace(0, len(y)/sr, len(y))
    ax.fill_between(times, y, alpha=0.7, color=PURPLE)
    ax.plot(times, y, color=PURPLE, linewidth=0.4, alpha=0.9)
    ax.axhline(0, color=DARK, linewidth=0.4, alpha=0.3)
    ax.set_xlim(0, times[-1])
    ax.set_xlabel("Time (s)", fontsize=8, color=DARK)
    ax.set_ylabel("Amplitude", fontsize=8, color=DARK)
    ax.tick_params(colors=DARK, labelsize=7)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(pad=0.4)
    return fig


def make_spectrogram_fig(y, sr):
    fig, ax = plt.subplots(figsize=(7, 2.2), facecolor=GRAY_BG)
    ax.set_facecolor(GRAY_BG)
    D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, hop_length=512,
                                   x_axis="time", y_axis="hz",
                                   ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    ax.set_xlabel("Time (s)", fontsize=8, color=DARK)
    ax.set_ylabel("Frequency (Hz)", fontsize=8, color=DARK)
    ax.tick_params(colors=DARK, labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#ccc")
    ax.set_title("STFT Spectrogram", fontsize=9, color=DARK, pad=4)
    fig.tight_layout(pad=0.4)
    return fig


def make_feature_radar(vec):
    groups = {
        "MFCCs":      slice(0, 26),
        "Centroid":   slice(26, 28),
        "Bandwidth":  slice(28, 30),
        "Rolloff":    slice(30, 32),
        "RMS":        slice(32, 34),
        "ZCR":        slice(34, 36),
    }
    labels = list(groups.keys())
    # normalise each group to 0-1 by L2 norm magnitude
    vals = []
    for sl in groups.values():
        chunk = vec[sl]
        norm = np.linalg.norm(chunk)
        vals.append(float(norm) if norm > 0 else 0.0)
    # scale to 0-1
    mx = max(vals) if max(vals) > 0 else 1
    vals = [v/mx for v in vals]

    N = len(labels)
    angles = [n/N*2*np.pi for n in range(N)] + [0]
    vals_plot = vals + [vals[0]]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True),
                           facecolor=GRAY_BG)
    ax.set_facecolor(GRAY_BG)
    ax.plot(angles, vals_plot, color=PURPLE, linewidth=2)
    ax.fill(angles, vals_plot, color=PURPLE, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color=DARK)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], size=6)
    ax.grid(color="#ccc", linewidth=0.5)
    ax.spines["polar"].set_color("#ccc")
    ax.set_title("Feature energy", size=9, color=DARK, pad=14)
    fig.tight_layout()
    return fig


def make_predictions_fig(predictions):
    """Bar chart of top-5 predictions per model."""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5*n_models, 4),
                             facecolor=GRAY_BG)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, preds) in zip(axes, predictions.items()):
        ax.set_facecolor(GRAY_BG)
        labels  = [p[0] for p in preds]
        scores  = [p[1] for p in preds]
        colors  = [MODEL_COLORS.get(model_name, PURPLE)] * len(labels)
        # highlight top
        colors[0] = MODEL_COLORS.get(model_name, PURPLE)
        alphas = [1.0 if i == 0 else 0.4 for i in range(len(labels))]

        bars = ax.barh(range(len(labels)), scores,
                       color=[MODEL_COLORS.get(model_name, PURPLE)] * len(labels),
                       alpha=0.85, height=0.6, edgecolor="none")
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5, color=DARK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence", fontsize=8, color=DARK)
        ax.set_title(model_name, fontsize=10, fontweight="bold",
                     color=MODEL_COLORS.get(model_name, DARK), pad=6)
        ax.tick_params(colors=DARK, labelsize=7)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.axvline(scores[0], color=MODEL_COLORS.get(model_name, PURPLE),
                   linewidth=0.8, linestyle="--", alpha=0.5)

        # confidence label
        ax.text(scores[0] + 0.02, 0, f"{scores[0]*100:.1f}%",
                va="center", fontsize=8, color=MODEL_COLORS.get(model_name, DARK),
                fontweight="bold")

    fig.suptitle("Top-5 predictions per model", fontsize=11,
                 color=DARK, y=1.01)
    fig.tight_layout(pad=0.6)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def classify_audio(audio_path, selected_models):
    if audio_path is None:
        return (None, None, None, None,
                "⚠ Please upload an audio file.", "")

    if not _models:
        status = load_or_train()
        if status == "no_data":
            return (None, None, None, None,
                    "❌ No trained models found. Run extract_features.py and train_models.py first.", "")

    # ── Extract features ──
    y, sr = load_audio(audio_path)
    vec   = extract_features(y, sr)

    # ── Run selected models ──
    if not selected_models:
        selected_models = list(_models.keys())

    predictions = {}
    summary_rows = []

    for mname in selected_models:
        if mname not in _models:
            continue
        pipe = _models[mname]

        # Get probabilities if available
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba([vec])[0]
            top_idx   = np.argsort(probs)[::-1][:5]
            top_preds = [(pipe.classes_[i], float(probs[i])) for i in top_idx]
        else:
            pred = pipe.predict([vec])[0]
            top_preds = [(pred, 1.0)]

        predictions[mname] = top_preds
        summary_rows.append({
            "Model":      mname,
            "Prediction": top_preds[0][0],
            "Confidence": f"{top_preds[0][1]*100:.1f}%",
        })

    # ── Build figures ──
    fig_wave  = make_waveform_fig(y, sr)
    fig_spec  = make_spectrogram_fig(y, sr)
    fig_radar = make_feature_radar(vec)
    fig_preds = make_predictions_fig(predictions) if predictions else None

    # ── Summary text ──
    lines = []
    for row in summary_rows:
        lines.append(f"**{row['Model']}** → `{row['Prediction']}` ({row['Confidence']})")
    summary_md = "\n\n".join(lines)

    # ── Feature table ──
    names = feature_names()
    feat_df = pd.DataFrame({
        "Feature": names,
        "Value":   [f"{v:.4f}" for v in vec],
    })
    feat_str = feat_df.to_string(index=False)

    return fig_wave, fig_spec, fig_radar, fig_preds, summary_md, feat_str


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'IBM Plex Mono', monospace; }
.gradio-container { max-width: 1100px; margin: auto; }
#title { text-align: center; padding: 1rem 0 0.4rem; }
#title h1 { font-size: 1.6rem; font-weight: 700; color: #534AB7; letter-spacing: -0.02em; }
#title p  { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
.result-label { font-size: 1rem !important; font-weight: 600 !important; }
"""

def build_ui():
    model_choices = list(_models.keys()) if _models else ["k-NN", "SVM (RBF)", "Random Forest"]

    with gr.Blocks(css=CSS, title="ESC-50 Classifier") as demo:

        gr.HTML("""
        <div id="title">
          <h1>Environmental Sound Classifier</h1>
          <p>Upload any audio clip → instant prediction from your trained ML models</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="Drop audio here (.wav / .mp3 / .ogg)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                model_sel = gr.CheckboxGroup(
                    choices=model_choices,
                    value=model_choices,
                    label="Models to run",
                )
                run_btn = gr.Button("Classify", variant="primary")

                summary_out = gr.Markdown(
                    label="Predictions",
                    elem_classes=["result-label"],
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Predictions"):
                        pred_plot = gr.Plot(label="Top-5 per model")
                    with gr.Tab("Waveform"):
                        wave_plot = gr.Plot(label="Audio waveform")
                    with gr.Tab("Spectrogram"):
                        spec_plot = gr.Plot(label="STFT spectrogram")
                    with gr.Tab("Feature radar"):
                        radar_plot = gr.Plot(label="Feature energy")
                    with gr.Tab("Raw features"):
                        feat_text = gr.Textbox(
                            label="All 36 feature values",
                            lines=20,
                            max_lines=40,
                        )

        run_btn.click(
            fn=classify_audio,
            inputs=[audio_in, model_sel],
            outputs=[wave_plot, spec_plot, radar_plot, pred_plot, summary_out, feat_text],
        )
        audio_in.change(
            fn=classify_audio,
            inputs=[audio_in, model_sel],
            outputs=[wave_plot, spec_plot, radar_plot, pred_plot, summary_out, feat_text],
        )

        gr.HTML("""
        <div style="text-align:center;padding:1rem 0 0.5rem;font-size:0.78rem;color:#999;">
          ELEC6651 · ESC-50 · k-NN / SVM / Random Forest
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Environmental Sound Classification Dashboard")
    print("=" * 55)

    status = load_or_train()
    if status == "loaded":
        print(f"✓ Models loaded  ({len(_models)} classifiers, {len(_classes)} classes)")
    elif status == "trained":
        print(f"✓ Models trained ({len(_models)} classifiers, {len(_classes)} classes)")
    else:
        print("⚠  No features.csv or trained_models.pkl found.")
        print("   Run your pipeline first:")
        print("   python src/extract_features.py --esc_root data/ESC-50")
        print("   python src/train_models.py")
        print()
        print("   Dashboard will still launch — upload audio to see")
        print("   feature plots even without predictions.\n")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )