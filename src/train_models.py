import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def plot_confusion(cm, labels, out_path, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", type=str, default="results/features.csv")
    parser.add_argument("--out_dir",      type=str, default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    y  = df["label"].values
    feature_cols  = [c for c in df.columns if c not in ["label", "fold", "filename"]]
    X             = df[feature_cols].values
    labels_sorted = sorted(df["label"].unique().tolist())

    models = {
        "knn": (
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7]},
        ),
        "svm_rbf": (
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
            {"clf__C": [1, 10], "clf__gamma": ["scale", 0.01]},
        ),
        "random_forest": (
            Pipeline([("clf", RandomForestClassifier(random_state=42))]),
            {"clf__n_estimators": [200], "clf__max_depth": [None, 20]},
        ),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (pipe, param_grid) in models.items():
        print(f"\n--- Training {name} ---")

        gs = GridSearchCV(pipe, param_grid, cv=skf, scoring="accuracy", n_jobs=-1)
        gs.fit(X, y)
        print(f"Best params: {gs.best_params_}")

        y_true_all, y_pred_all = [], []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_model = gs.best_estimator_
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        acc = accuracy_score(y_true_all, y_pred_all)
        print(f"Cross-validated accuracy: {acc:.4f} ({acc*100:.1f}%)")

        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)
        cm_path = os.path.join(args.out_dir, f"confusion_{name}.png")
        plot_confusion(cm, labels_sorted, cm_path, f"Confusion Matrix - {name}")

        report_str = classification_report(
            y_true_all, y_pred_all,
            target_names=labels_sorted,
            digits=4
        )
        print(f"\nClassification report ({name}):\n{report_str}")

        report_dict = classification_report(
            y_true_all, y_pred_all,
            target_names=labels_sorted,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv_path = os.path.join(args.out_dir, f"report_{name}.csv")
        report_df.to_csv(report_csv_path, float_format="%.4f")

        report_txt_path = os.path.join(args.out_dir, f"report_{name}.txt")
        with open(report_txt_path, "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Best params: {gs.best_params_}\n")
            f.write(f"Overall accuracy: {acc:.4f}\n\n")
            f.write(report_str)

        results[name] = {
            "best_params":      gs.best_params_,
            "accuracy":         float(acc),
            "confusion_matrix": cm_path,
            "report_csv":       report_csv_path,
            "report_txt":       report_txt_path,
        }

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete. All results saved in '{args.out_dir}/'")


if __name__ == "__main__":
    main()