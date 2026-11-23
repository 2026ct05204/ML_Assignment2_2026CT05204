from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "bank.csv"
MODEL_DIR = ROOT / "model"
RESULT_DIR = ROOT / "results"
RANDOM_STATE = 42

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Load Kaggle bank-full.csv; Kaggle's original file is semicolon-separated."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download bank.csv from the Kaggle link in README.md "
            "and place it in the dataset folder."
        )

    # sep=None attempts delimiter detection; the explicit fallback is ';'.
    try:
        df = pd.read_csv(DATA_PATH, sep=None, engine="python")
        if len(df.columns) == 1:
            df = pd.read_csv(DATA_PATH, sep=";")
    except Exception:
        df = pd.read_csv(DATA_PATH, sep=";")

    # Remove accidental spaces from column names and string values.
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def inspect_dataset(df):
    print("\n========== DATASET OVERVIEW ==========")
    print("Shape (rows, columns):", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values before handling:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    if "deposit" not in df.columns:
        raise ValueError("Expected target column 'deposit' was not found.")

    print("\nTarget distribution:\n", df["deposit"].value_counts(dropna=False))
    print("\nTarget percentages:\n",
          (df["deposit"].value_counts(normalize=True, dropna=False) * 100).round(2))


def prepare_data(df):
    # Remove exact duplicate records before splitting.
    df = df.drop_duplicates().copy()

    # Exclude duration because the call duration is known only after the
    # marketing call is completed.
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Treat the dataset's 'unknown' category as unavailable information.
    # The imputer below learns replacement values from training data only.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("unknown", np.nan)

    y = df["deposit"].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Target 'deposit' contains unexpected values. Inspect the target.")

    X = df.drop(columns=["deposit"])

    # Identify columns by their data type.
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Stratification maintains the class proportions in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Fit imputers, encoders and scaler inside each model Pipeline.
    # This prevents preprocessing from learning from the test set.
    numeric_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Force dense output because GaussianNB requires dense arrays.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_features),
            ("cat", categorical_preprocess, categorical_features)
        ],
        sparse_threshold=0
    )

    # Keep a copy of the held-out test data and target for reproducibility.
    test_export = X_test.copy()
    test_export["target_y"] = y_test.map({0: "no", 1: "yes"})
    test_export.to_csv(ROOT / "test_data.csv", index=False)

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training target proportions:\n", y_train.value_counts(normalize=True).round(4))
    print("Testing target proportions:\n", y_test.value_counts(normalize=True).round(4))

    return X_train, X_test, y_train, y_test, preprocessor


def build_models(preprocessor):
    # Model settings are selected to provide a reasonable comparison 
    # between five different classification algorithms. 
    # Class weighting is used in Logistic regression, Decision tree and 
    #random forests to account for the class distribution. 
    #Maximum iteration are increased for logistic regression.
    #Tree depth is limited to comtrol complexity. KNN uses 
    #seven neighbours to classify observations based on nearby datapoints 
    #Gaussian naive bayes uses its standard configuration. Random forests uses 200 trees
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    pipelines = {}
    for name, classifier in classifiers.items():
        pipelines[name] = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ])
    return pipelines


def evaluate_models(pipelines, X_train, X_test, y_train, y_test):
    rows = []

    for name, pipeline in pipelines.items():
        print(f"\nTraining: {name}")
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, probabilities),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions)
        }
        rows.append(row)

        # Save each fitted preprocessing + classifier pipeline.
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.joblib")
        print(pd.Series(row).to_string())

    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv(RESULT_DIR / "model_comparison.csv")
    return results


def create_graphs(results):
    ax = results.plot(kind="bar", figsize=(13, 7), ylim=(-1, 1.05))
    ax.set_title("Classification Model Performance Comparison")
    ax.set_ylabel("Metric score")
    ax.set_xlabel("Machine learning model")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "model_comparison.png", dpi=200)
    plt.close()

    # Separate plot makes the metric comparison easier to read.
    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for metric in metrics:
        plt.figure(figsize=(9, 5))
        results[metric].sort_values(ascending=False).plot(kind="bar")
        plt.title(f"{metric} by Model")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"{metric.lower()}_comparison.png", dpi=180)
        plt.close()


def main():
    df = load_dataset()
    inspect_dataset(df)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    pipelines = build_models(preprocessor)
    results = evaluate_models(pipelines, X_train, X_test, y_train, y_test)
    create_graphs(results)

    print("\n========== FINAL COMPARISON ==========")
    print(results.round(4))
    print("\nFiles created:")
    print("- test_data.csv")
    print("- results/model_comparison.csv")
    print("- results/model_comparison.png")
    print("- results/*_comparison.png")
    print("- model/*.joblib")
    print("\nInterpret the actual metrics before naming the overall winner.")


if __name__ == "__main__":
    main()
