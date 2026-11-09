"""Shared, side-effect-free data, formatting and classification calculations."""
from pathlib import Path
import csv
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = ["customer_tenure_months", "complaint_count_90d", "discount_order_share",
            "avg_order_value_inr_hundreds", "loyalty_score", "cod_flag", "plus_member_flag"]
DATA_MD5 = "c6f249ba693f1ebeed318d6d5e18de23"


def load_data():
    path = ROOT / "data" / "records.csv"
    if hashlib.md5(path.read_bytes()).hexdigest() != DATA_MD5:
        raise ValueError("Dataset differs from the prescribed file; rerun generate_data.py")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    y = np.array([int(r["churned"]) for r in rows])
    value = np.array([float(r["order_value_inr_hundreds"]) for r in rows])
    return x, y, value


def standardise(x, mean=None, sd=None):
    if mean is None:
        mean = x.mean(axis=0)
        sd = np.sqrt(np.sum((x - mean)**2, axis=0) / (len(x) - 1))
    return (x - mean) / sd, mean, sd


def split_indices(y):
    train = np.zeros(len(y), dtype=bool)
    for outcome in (0, 1):
        indices = np.flatnonzero(y == outcome)
        train[indices[:7 * len(indices) // 10]] = True
    return np.flatnonzero(train), np.flatnonzero(~train)


def classification_data():
    x, y, _ = load_data()
    tr, te = split_indices(y)
    train, mean, sd = standardise(x[tr])
    test, _, _ = standardise(x[te], mean, sd)
    return x, y, tr, te, train, test


def sigmoid(z):
    # Separate branches avoid overflow without clipping probabilities.
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1 + ez)
    return out


def bce(x, y, w, b):
    z = x @ w + b
    # Algebraically identical to BCE, stable even for extreme logits.
    return float(np.mean(np.logaddexp(0, z) - y * z))


def logistic_fit(x, y, eta=0.1, updates=1000):
    w, b = np.zeros(x.shape[1]), 0.0
    losses = {0: bce(x, y, w, b)}
    for epoch in range(1, updates + 1):
        error = sigmoid(x @ w + b) - y
        # Both gradients use the same pre-update state.
        w -= eta * (x.T @ error) / len(y)
        b -= eta * float(error.mean())
        if epoch in (1, 10, 100, updates):
            losses[epoch] = bce(x, y, w, b)
    return w, b, losses


def metrics(y, predicted):
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    tn = int(np.sum(~predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    def ratio(a, b):
        return a / b if b else 0.0
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=ratio(tp + tn, len(y)), precision=precision, recall=recall,
                f1=ratio(2 * precision * recall, precision + recall),
                specificity=ratio(tn, tn + fp), tpr=recall, fpr=ratio(fp, fp + tn))


def knn_neighbours(reference, labels, queries, max_k):
    distances = np.sqrt(np.sum((queries[:, None, :] - reference[None, :, :])**2, axis=2))
    order = np.argsort(distances, axis=1, kind="stable")[:, :max_k]
    return labels[order]


def knn_select(train, labels):
    ks = list(range(1, 22, 2))
    scores = np.zeros((5, len(ks)))
    folds = np.arange(len(labels)) % 5
    for fold in range(5):
        held = folds == fold
        neighbours = knn_neighbours(train[~held], labels[~held], train[held], max(ks))
        counts = np.cumsum(neighbours, axis=1)
        for j, k in enumerate(ks):
            scores[fold, j] = metrics(labels[held], counts[:, k-1] >= (k+1)//2)["f1"]
    means = scores.mean(axis=0)
    # argmax returns the first maximum, giving the smaller K on ties.
    return ks[int(np.argmax(means))], list(zip(ks, means))


def cost(y, probabilities, tau):
    m = metrics(y, probabilities >= tau)
    return 50 * m["fp"] + 400 * m["fn"]


def formatted(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def write_results(part, results):
    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True)
    with (folder / f"part{part}_results.txt").open("w", encoding="utf-8", newline="\n") as f:
        for key, value in results.items():
            f.write(f"{key} = {formatted(value)}\n")


def write_csv(name, header, rows):
    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True)
    with (folder / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header.split(","))
        writer.writerows([[formatted(v) for v in row] for row in rows])
