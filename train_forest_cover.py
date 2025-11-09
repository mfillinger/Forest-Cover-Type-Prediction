# CSI 5810 - Forest Cover Type Prediction Project


import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import joblib


def load_dataset():
    """
    Load Kaggle Forest Cover Type training data.
    Expects train.csv in the same folder with columns:
    Id, features..., Cover_Type.
    """
    path = Path("train.csv")
    if not path.exists():
        raise FileNotFoundError("train.csv not found. Make sure it is in the same folder as this script.")

    df = pd.read_csv(path)

    if "Cover_Type" not in df.columns:
        raise ValueError("Cover_Type column not found in train.csv. Check that you have the correct file.")

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]

    print("[INFO] Loaded train.csv")
    print(f"[INFO] Shape: {X.shape}, Number of classes: {y.nunique()}")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing:
    - Standardize continuous features
    - Keep binary indicator features (e.g., Wilderness_Area, Soil_Type) as-is
    Binary features detected as columns with only {0,1}.
    """
    binary_cols = []
    cont_cols = []

    for c in X.columns:
        vals = set(X[c].unique())
        if vals.issubset({0, 1}) and len(vals) <= 2:
            binary_cols.append(c)
        else:
            cont_cols.append(c)

    print(f"[INFO] Continuous features: {len(cont_cols)}")
    print(f"[INFO] Binary features: {len(binary_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), cont_cols),
            ("bin", "passthrough", binary_cols),
        ]
    )

    return preprocessor


def evaluate_models(X, y, preprocessor):
    """
    Evaluate:
      - kNN (k=7)
      - Logistic Regression (multinomial)
      - Gaussian Naive Bayes
      - Linear Discriminant Analysis
    using 10-fold stratified CV.
    Return best model pipeline and metrics.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model_defs = {
        "kNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=500,
            n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis()
    }

    results = {}

    for name, clf in model_defs.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", clf)
        ])

        print(f"\n[INFO] Evaluating {name} with 10-fold stratified CV...")
        acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        f1_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

        results[name] = {
            "acc_mean": float(acc_scores.mean()),
            "acc_std": float(acc_scores.std()),
            "f1_mean": float(f1_scores.mean()),
            "f1_std": float(f1_scores.std())
        }

        print(
            f"[RESULT] {name} | "
            f"Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f} | "
            f"Weighted F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
        )

    best_name = max(results, key=lambda m: results[m]["acc_mean"])
    print(f"\n[INFO] Best model by CV accuracy: {best_name}")

    best_clf = model_defs[best_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", best_clf)
    ])

    print("[INFO] Fitting best model on training split...")
    best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    holdout_acc = accuracy_score(y_test, y_pred)
    holdout_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n[HOLD-OUT PERFORMANCE]")
    print(f"Accuracy: {holdout_acc:.4f}")
    print(f"Weighted F1: {holdout_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return best_name, best_pipe, results


def main():
    X, y = load_dataset()
    preprocessor = build_preprocessor(X)
    best_name, best_model, results = evaluate_models(X, y, preprocessor)

    joblib.dump(best_model, "best_forest_model.pkl")
    print(f"\n[SAVED] best_forest_model.pkl ({best_name})")

    results_df = pd.DataFrame(results).T
    results_df.to_csv("model_comparison_results.csv", index=True)
    print("[SAVED] model_comparison_results.csv")


if __name__ == "__main__":
    main()
