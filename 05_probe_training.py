import h5py
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

MODELS = {
    "mistral-7b-v03": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b-it",
        "num_layers": 42,
        "num_heads": 16,
        "head_dim": 256,
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
    },
}

def load_attention_outputs(model_key, path):
    with h5py.File(path, "r") as f:
        att = f["attention"][:]
        target = f["target_label"][:].astype(int)
        split = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in f["split"][:]])

    num_layers = MODELS[model_key]["num_layers"]
    num_heads = MODELS[model_key]["num_heads"]
    head_dim = MODELS[model_key]["head_dim"]

    num_examples = att.shape[0]
    X = att.reshape(num_examples, num_layers * num_heads * head_dim)
    train_idx = np.where(split == "train")[0]
    validate_idx = np.where(split == "validate")[0]
    test_idx = np.where(split == "test")[0]
    X_train, X_validate, X_test = X[train_idx], X[validate_idx], X[test_idx]
    y_train, y_validate, y_test = target[train_idx], target[validate_idx], target[test_idx]
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def structure_features(model_key):
    num_layers = MODELS[model_key]["num_layers"]
    num_heads = MODELS[model_key]["num_heads"]
    head_dim = MODELS[model_key]["head_dim"]
    
    head_slices = {}

    for l in range(num_layers):
        for h in range(num_heads):
            start = (l * num_heads + h) * head_dim
            head_slices[(l, h)] = slice(start, start + head_dim)

    return head_slices

def train_and_validate(X_train, y_train, X_validate, y_validate):
    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X_train), y_train)

    preds = clf.predict(sc.transform(X_validate))
    return accuracy_score(y_validate, preds)

def evaluate(clf, sc, X, y):
    Xs = sc.transform(X)
    probs = clf.predict_proba(Xs)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y, probs),
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds)
    }

def best_heads(acc_df, top_k):
    sorted_df = acc_df.sort_values(by="accuracy", ascending=False)
    top_heads = sorted_df.head(top_k)

    return top_heads[["layer", "head"]].apply(tuple, axis=1).tolist()


def train_eval_top_heads(X_train, X_test, y_train, y_test, head_slices, heads):
    slcs = [head_slices[(l, h)] for (l, h) in heads]
    full_slc = np.hstack([np.arange(s.start, s.stop) for s in slcs])

    X_train_sl = X_train[:, full_slc]
    X_test_sl = X_test[:, full_slc]

    sc = StandardScaler().fit(X_train_sl)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(X_train_sl), y_train)
    metrics = evaluate(clf, sc, X_test_sl, y_test)
    metrics["selected_heads"] = [f"L{int(h[0])}, H{int(h[1])}" for h in heads]
    return clf, sc, metrics, full_slc


def main():
    final_evals = []
    for model_key in MODELS:
        for mode in [
            "with_user_question",
            "without_user_question",
        ]:
            print(f"Running: {model_key}, mode: {mode}")

            path = DATA_DIR / "attention_outputs" / f"{model_key}_attention_outputs_{mode}.h5"

            X_train, X_validate, X_test, y_train, y_validate, y_test = load_attention_outputs(model_key, path)

            num_layers = MODELS[model_key]["num_layers"]
            num_heads = MODELS[model_key]["num_heads"]
            head_slices = structure_features(model_key)

            acc_results = []
            for l in range(num_layers):
                print(f"Evaluating heads for layer {l}/{num_layers}")
                for h in range(num_heads):
                    slc = head_slices[(l, h)]
                    acc = train_and_validate(
                        X_train[:, slc], y_train, X_validate[:, slc], y_validate
                    )
                    acc_results.append({"layer": l, "head": h, "accuracy": acc})

            acc_df = pd.DataFrame(acc_results)
            acc_path = DATA_DIR / "probe_evaluations" / f"{model_key}_{mode}_per_head_accuracy.csv"
            acc_path.parent.mkdir(parents=True, exist_ok=True)
            acc_df.to_csv(acc_path, index=False)

            print(f"Training final logistic regression probe for {model_key}")
            heads = best_heads(acc_df, top_k=10)

            clf, sc, metrics, slc = train_eval_top_heads(
                X_train, X_test, y_train, y_test, head_slices, heads
            )

            result_row = {"model": model_key, "mode": mode, **metrics}
            final_evals.append(result_row)

            probe_path = DATA_DIR / "trained_probes" / f"{model_key}_{mode}_final_probe.pkl"
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(probe_path, "wb") as f:
                pickle.dump(dict(clf=clf, scaler=sc, heads=heads, slice=slc), f)

    eval_df = pd.DataFrame(final_evals)
    out_path = DATA_DIR / "probe_evaluations" / "final_probe_evaluations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()