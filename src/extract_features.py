import os
import argparse
import pandas as pd
from tqdm import tqdm
from utils import load_audio, extract_feature_vector, feature_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="results/features.csv")
    args = parser.parse_args()

    meta_path = os.path.join(args.esc_root, "meta", "esc50.csv")
    audio_dir = os.path.join(args.esc_root, "audio")

    meta = pd.read_csv(meta_path)

    X, y, folds, filenames = [], [], [], []
    col_names = feature_names()

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        path = os.path.join(audio_dir, row["filename"])
        audio, sr = load_audio(path)
        vec = extract_feature_vector(audio, sr)
        X.append(vec)
        y.append(row["category"])
        folds.append(row["fold"])
        filenames.append(row["filename"])

    df = pd.DataFrame(X, columns=col_names)
    df["label"] = y
    df["fold"] = folds
    df["filename"] = filenames

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved features to", args.out_csv)

if __name__ == "__main__":
    main()