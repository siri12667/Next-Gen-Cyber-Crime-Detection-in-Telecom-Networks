"""
Core machine-learning pipeline for Next-Gen Cyber Crime Detection in Telecom Networks.

Stages (match the System Architecture slide):
    Data Collection -> Preprocessing -> Feature Extraction -> ML Module -> Prediction
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

LABEL_COL = "label"
# Columns that leak the answer or are only identifiers
DROP_COLS = ["id", "attack_cat"]
RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    model: object
    precision: float
    recall: float
    f1: float
    accuracy: float
    report: str
    confusion: np.ndarray

    def summary(self) -> str:
        return (f"{self.name} Precision : {self.precision * 100:.4f}\n"
                f"{self.name} Recall    : {self.recall * 100:.4f}\n"
                f"{self.name} FMeasure  : {self.f1 * 100:.4f}\n"
                f"{self.name} Accuracy  : {self.accuracy * 100:.4f}\n\n"
                f"Classification Report:\n{self.report}\n"
                f"Confusion Matrix:\n{self.confusion}\n")


@dataclass
class PipelineState:
    raw: Optional[pd.DataFrame] = None
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    feature_names: list = field(default_factory=list)
    encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    results: Dict[str, ModelResult] = field(default_factory=dict)


class CyberCrimePipeline:
    def __init__(self):
        self.s = PipelineState()

    # ------------------------------------------------------------------ 1. load
    def load_dataset(self, path: str) -> str:
        df = pd.read_csv(path)
        if LABEL_COL not in df.columns:
            raise ValueError(f"Dataset must contain a '{LABEL_COL}' column "
                             f"(0 = normal traffic, 1 = attack).")
        self.s = PipelineState(raw=df)
        n_att = int((df[LABEL_COL] == 1).sum())
        return (f"Dataset loaded: {path}\n"
                f"Rows: {len(df)}   Columns: {df.shape[1]}\n"
                f"Normal: {len(df) - n_att}   Attack: {n_att}\n")

    # ----------------------------------------------------------- 2. preprocess
    def preprocess(self) -> str:
        if self.s.raw is None:
            raise RuntimeError("Upload a dataset first.")
        df = self.s.raw.copy()
        before = len(df)
        n_missing = int(df.isna().sum().sum())

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        df = df.drop_duplicates()
        n_dups = before - len(df)

        # fill missing values: median for numeric, mode for categorical
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype.kind in "biuf":
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

        y = df[LABEL_COL].astype(int)
        X = df.drop(columns=[LABEL_COL])

        # encode text columns (proto, service, state, ...)
        encoders = {}
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        self.s.X, self.s.y = X, y
        self.s.encoders = encoders
        self.s.feature_names = list(X.columns)
        return (f"Preprocessing complete.\n"
                f"Missing values filled : {n_missing}\n"
                f"Duplicate rows removed: {n_dups}\n"
                f"Categorical columns encoded: {list(encoders) or 'none'}\n"
                f"Final feature count   : {X.shape[1]}   Rows: {len(X)}\n")

    # -------------------------------------------------------------- 3. splitting
    def split(self, test_size: float = 0.2) -> str:
        if self.s.X is None:
            raise RuntimeError("Preprocess the dataset first.")
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.s.X.values, self.s.y.values, test_size=test_size,
            random_state=RANDOM_STATE, stratify=self.s.y.values)
        self.s.scaler = StandardScaler().fit(X_tr)
        self.s.X_train, self.s.X_test = X_tr, X_te
        self.s.y_train, self.s.y_test = y_tr, y_te
        return (f"Data split done ({int((1 - test_size) * 100)}% train / {int(test_size * 100)}% test).\n"
                f"Training rows: {len(X_tr)}   Testing rows: {len(X_te)}\n")

    # ----------------------------------------------------------- 4/5. training
    def _evaluate(self, name, model, X_test) -> ModelResult:
        pred = model.predict(X_test)
        y = self.s.y_test
        return ModelResult(
            name=name, model=model,
            precision=precision_score(y, pred, average="macro", zero_division=0),
            recall=recall_score(y, pred, average="macro", zero_division=0),
            f1=f1_score(y, pred, average="macro", zero_division=0),
            accuracy=accuracy_score(y, pred),
            report=classification_report(y, pred, zero_division=0),
            confusion=confusion_matrix(y, pred))

    def _need_split(self):
        if self.s.X_train is None:
            raise RuntimeError("Run 'Data splitting' first.")

    def train_logistic_regression(self) -> ModelResult:
        self._need_split()
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(self.s.scaler.transform(self.s.X_train), self.s.y_train)
        res = self._evaluate("Logistic Regression", model,
                             self.s.scaler.transform(self.s.X_test))
        self.s.results["lr"] = res
        return res

    def train_random_forest(self) -> ModelResult:
        self._need_split()
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                       random_state=RANDOM_STATE)
        model.fit(self.s.X_train, self.s.y_train)
        res = self._evaluate("Random Forest", model, self.s.X_test)
        self.s.results["rf"] = res
        return res

    # ---------------------------------------------------------- 6. prediction
    def predict_rows(self, n: int = 10) -> str:
        """Predict the first n test rows with the best trained model."""
        if not self.s.results:
            raise RuntimeError("Train at least one model first.")
        best_key = max(self.s.results, key=lambda k: self.s.results[k].accuracy)
        best = self.s.results[best_key]
        X = self.s.X_test[:n]
        Xin = self.s.scaler.transform(X) if best_key == "lr" else X
        preds = best.model.predict(Xin)
        lines = [f"Predictions using best model: {best.name} "
                 f"(accuracy {best.accuracy * 100:.2f}%)\n"]
        for i, (p, actual) in enumerate(zip(preds, self.s.y_test[:n]), 1):
            verdict = "Cyber attack detected" if p == 1 else "No Cyber Attack detected"
            truth = "attack" if actual == 1 else "normal"
            lines.append(f"Predicted output for row {i}: {verdict}   (actual: {truth})")
        return "\n".join(lines) + "\n"

    def predict_new(self, df: pd.DataFrame) -> np.ndarray:
        """Predict raw rows (same columns as the training CSV) -> array of 0/1."""
        if not self.s.results:
            raise RuntimeError("Train at least one model first.")
        best_key = max(self.s.results, key=lambda k: self.s.results[k].accuracy)
        X = df.drop(columns=[c for c in DROP_COLS + [LABEL_COL] if c in df.columns]).copy()
        for col, le in self.s.encoders.items():
            known = set(le.classes_)
            X[col] = le.transform(X[col].astype(str).where(X[col].astype(str).isin(known), le.classes_[0]))
        X = X[self.s.feature_names].values
        if best_key == "lr":
            X = self.s.scaler.transform(X)
        return self.s.results[best_key].model.predict(X)

    # ----------------------------------------------------------- 7. comparison
    def comparison_figure(self):
        import matplotlib
        from matplotlib.figure import Figure
        if not self.s.results:
            raise RuntimeError("Train at least one model first.")
        metrics = ["Precision", "Recall", "F1-score", "Accuracy"]
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        n = len(self.s.results)
        width = 0.8 / n
        for i, res in enumerate(self.s.results.values()):
            vals = [res.precision * 100, res.recall * 100, res.f1 * 100, res.accuracy * 100]
            xs = np.arange(len(metrics)) + i * width
            bars = ax.bar(xs, vals, width, label=res.name)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                        ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(metrics)) + width * (n - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.set_title("Logistic Regression vs Random Forest - Cyber Attack Detection")
        ax.legend()
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------- save / load
    def save_models(self, folder: str = "models") -> str:
        Path(folder).mkdir(exist_ok=True)
        joblib.dump({"results": {k: v.model for k, v in self.s.results.items()},
                     "encoders": self.s.encoders, "scaler": self.s.scaler,
                     "features": self.s.feature_names},
                    Path(folder) / "cyber_models.joblib")
        return str(Path(folder) / "cyber_models.joblib")
