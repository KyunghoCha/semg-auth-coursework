"""Step 1: trial-safe CWT, three CNNs, repeated holdout and DenseNet CV."""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/semg-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, welch
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import densenet161, resnet18

from data_utils import DATA_COMMIT, LABELS, load_trials, split_trials

ROOT = Path(__file__).resolve().parent
MODELS = ["SimpleCNN", "ResNet18", "DenseNet161"]
WIN, HOP, WINDOWS_PER_TRIAL = 300, 150, 19


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preprocess(signal):
    """Reapply the lecture filters to the already-filtered public recording."""
    bn, an = iirnotch(60, 30, fs=1000)
    b, a = butter(4, [20, 499], btype="bandpass", fs=1000)
    return filtfilt(b, a, filtfilt(bn, an, signal, axis=0), axis=0)


def make_windows(signal):
    return np.stack([signal[start:start + WIN] for start in range(0, len(signal) - WIN + 1, HOP)])


def normalize_windows(windows):
    minimum = windows.min(axis=(1, 2), keepdims=True)
    maximum = windows.max(axis=(1, 2), keepdims=True)
    return (windows - minimum) / (maximum - minimum + 1e-8)


def to_cwt(window):
    maps = [np.abs(pywt.cwt(window[:, channel], np.arange(1, 33), "morl")[0])
            for channel in range(2)]
    return np.stack([maps[0], maps[1], (maps[0] + maps[1]) / 2]).astype(np.float32)


def model_for(name):
    if name == "SimpleCNN":
        return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 5))
    if name == "ResNet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 5)
        return model
    if name == "DenseNet161":
        model = densenet161(weights=None)
        model.classifier = nn.Linear(2208, 5)
        return model
    raise ValueError(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrialWindows(Dataset):
    def __init__(self, cache, trials, indices):
        self.cache = cache
        self.trials = trials
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices) * WINDOWS_PER_TRIAL

    def __getitem__(self, index):
        trial_id = self.indices[index // WINDOWS_PER_TRIAL]
        window_id = index % WINDOWS_PER_TRIAL
        # Copy a read-only memory map slice; PyTorch must not mutate the cache.
        return torch.from_numpy(np.array(self.cache[trial_id, window_id], copy=True)), LABELS.index(self.trials[trial_id].subject)


def make_plan(trials):
    train, test = split_trials(trials, test_size=0.2, seed=42)
    labels = np.array([trial.subject for trial in trials])
    folds = []
    # Keep the final holdout entirely outside cross-validation.
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, labels[train]), 1):
        folds.append({"fold": fold, "train": train[tr].tolist(), "test": train[va].tolist()})
    return {"holdout": {"train": train.tolist(), "test": test.tolist()}, "cv": folds}


def prepare_cache(trials, plan, cache_dir, results_dir):
    """A complete trial split plan is saved before any overlapping windows exist."""
    signature = hashlib.sha256(json.dumps({"source": DATA_COMMIT,
        "hashes": [trial.signal_sha256 for trial in trials], "filter": [60, 30, 20, 499, 4],
        "win": WIN, "hop": HOP, "wavelet": "morl", "scales": 32}, sort_keys=True).encode()).hexdigest()
    cache_file = cache_dir / "cwt.npy"
    meta_file = cache_dir / "cwt_metadata.json"
    expected_shape = (len(trials), WINDOWS_PER_TRIAL, 3, 32, WIN)
    if cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("signature") != signature:
            raise ValueError("Existing CWT cache uses different data or preprocessing")
        cached = np.load(cache_file, mmap_mode="r")
        if cached.shape != expected_shape or cached.dtype != np.float32:
            raise ValueError("Unexpected CWT cache shape or dtype")
        return cached
    started = time.perf_counter()
    cache = np.lib.format.open_memmap(cache_file, mode="w+", dtype=np.float32, shape=expected_shape)
    # Generate training windows first, then held-out windows, without shared statistics.
    order = plan["holdout"]["train"] + plan["holdout"]["test"]
    for position, index in enumerate(order, 1):
        windows = normalize_windows(make_windows(preprocess(trials[index].signal)))
        cache[index] = np.stack([to_cwt(window) for window in windows])
        if position % 50 == 0:
            print(f"CWT {position}/{len(trials)}", flush=True)
    cache.flush()
    save_json(meta_file, {"signature": signature, "shape": list(expected_shape),
                          "seconds": time.perf_counter() - started})
    first = trials[plan["holdout"]["train"][0]]
    filtered = preprocess(first.signal)
    f0, p0 = welch(first.signal[:, 0], fs=1000, nperseg=512)
    f1, p1 = welch(filtered[:, 0], fs=1000, nperseg=512)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.semilogy(f0, p0, label="Provided filtered CSV")
    ax.semilogy(f1, p1, label="After additional lecture filters")
    ax.axvline(60, color="red", linestyle="--", linewidth=.8)
    ax.set(xlim=(0, 300), xlabel="Frequency (Hz)", ylabel="Power", title=first.path)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(results_dir / "filter_check.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    im = ax.imshow(cache[plan["holdout"]["train"][0], 0, 0], aspect="auto", origin="lower",
                   extent=[0, .3, 1, 32], cmap="viridis")
    ax.set(xlabel="Time (s)", ylabel="CWT scale", title="First training window, channel 1")
    fig.colorbar(im, ax=ax, label="Absolute coefficient"); fig.tight_layout()
    fig.savefig(results_dir / "cwt_example.png", dpi=150); plt.close(fig)
    del cache
    return np.load(cache_file, mmap_mode="r")


def train_run(name, seed, train_indices, test_indices, cache, trials, epochs, batch_size, threads, output, save_weights):
    complete = output / "metrics.json"
    config = {"model": name, "seed": seed, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": .001, "optimizer": "Adam", "pretrained": False,
              "train_indices": list(train_indices), "test_indices": list(test_indices), "threads": threads,
              "code_sha256": {filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
                              for filename in ["experiment.py", "data_utils.py"]}}
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["config"] != config:
            raise ValueError(f"Completed run has different settings: {output}")
        print(f"Resume: completed {output.name}", flush=True)
        return saved
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    train_data = TrialWindows(cache, trials, train_indices)
    test_data = TrialWindows(cache, trials, test_indices)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model_for(name)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    began = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.; train_correct = 0
        epoch_start = time.perf_counter()
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.numel()
            train_correct += int((logits.argmax(1) == labels).sum())
        row = {"epoch": epoch, "train_loss": total_loss / len(train_data),
               "train_accuracy": train_correct / len(train_data), "seconds": time.perf_counter() - epoch_start}
        losses.append(row)
        write_csv(output / "training.csv", losses)
        print(f"{output.name} epoch {epoch}/{epochs}: loss={row['train_loss']:.5f} acc={row['train_accuracy']:.4f} seconds={row['seconds']:.1f}", flush=True)
    train_seconds = time.perf_counter() - began
    model.eval(); predicted, actual = [], []
    with torch.inference_mode():
        for features, labels in test_loader:
            predicted.extend(model(features).argmax(1).tolist()); actual.extend(labels.tolist())
        sample = test_data[0][0].unsqueeze(0)
        for _ in range(10): model(sample)
        start = time.perf_counter()
        for _ in range(100): model(sample)
        inference_ms = (time.perf_counter() - start) * 1000 / 100
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, labels=range(5), average="macro", zero_division=0)
    cm = confusion_matrix(actual, predicted, labels=range(5))
    prediction_rows = []
    for offset, (truth, prediction) in enumerate(zip(actual, predicted)):
        trial = trials[test_indices[offset // WINDOWS_PER_TRIAL]]
        prediction_rows.append({"path": trial.path, "window": offset % WINDOWS_PER_TRIAL,
                                "true": LABELS[truth], "predicted": LABELS[prediction]})
    write_csv(output / "predictions.csv", prediction_rows)
    result = {"config": config, "n_train_trials": len(train_indices), "n_test_trials": len(test_indices),
              "n_train_windows": len(train_data), "n_test_windows": len(test_data),
              "accuracy": float(accuracy_score(actual, predicted)), "precision_macro": float(precision),
              "recall_macro": float(recall), "f1_macro": float(f1), "confusion_matrix": cm.tolist(),
              "train_seconds": train_seconds, "parameters": parameters, "inference_ms_per_window": inference_ms,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    if save_weights:
        torch.save(model.state_dict(), ROOT / "cache" / f"{name}_seed42.pt")
    save_json(complete, result)
    del model, optimizer, train_loader, test_loader
    gc.collect()
    print(f"Completed {output.name}: accuracy={result['accuracy']:.4f}, macro_f1={f1:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--phase", choices=["all", "holdout", "cv"], default="all")
    args = parser.parse_args()
    if args.epochs < 5 or args.threads < 1:
        parser.error("At least five epochs and one CPU thread are required")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    results = ROOT / "results"; results.mkdir(exist_ok=True)
    cache_dir = ROOT / "cache"; cache_dir.mkdir(exist_ok=True)
    trials, duplicates = load_trials(ROOT / "data" / "data")
    plan = make_plan(trials)
    save_json(results / "splits.json", plan)
    train_set = set(plan["holdout"]["train"])
    write_csv(results / "manifest.csv", [{"index": i, "path": trial.path, "subject": trial.subject,
              "trial": trial.number, "sha256": trial.sha256, "signal_sha256": trial.signal_sha256,
              "holdout_split": "train" if i in train_set else "test"} for i, trial in enumerate(trials)])
    save_json(results / "protocol.json", {"source": "https://github.com/sea3551/palm-sEMG-doorknob-filtered",
        "data_commit": DATA_COMMIT, "source_file_count": 250, "unique_trials": len(trials), "duplicates_removed": duplicates,
        "split_seed": 42, "holdout_train_seeds": [42, 43, 44], "cv_scope": "holdout training trials only",
        "epochs": args.epochs, "batch_size": 16, "threads": args.threads, "device": "cpu", "filter_reapplied": True,
        "window": WIN, "hop": HOP, "normalization": "minmax per window across time and both channels, eps=1e-8",
        "wavelet": "morl", "scales": list(range(1, 33)), "input_shape": [3, 32, 300],
        "third_channel": "mean of two absolute CWT maps", "metric_unit": "window", "metric_average": "macro",
        "versions": {"python": os.sys.version.split()[0], "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "torch": torch.__version__, "torchvision": torchvision.__version__, "PyWavelets": pywt.__version__},
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["experiment.py", "data_utils.py"]}})
    cache = prepare_cache(trials, plan, cache_dir, results)
    if args.phase in ["all", "holdout"]:
        for name in MODELS:
            for seed in [42, 43, 44]:
                train_run(name, seed, plan["holdout"]["train"], plan["holdout"]["test"], cache, trials,
                          args.epochs, 16, args.threads, results / "runs" / f"holdout_{name}_seed{seed}", seed == 42)
    if args.phase in ["all", "cv"]:
        for fold in plan["cv"]:
            train_run("DenseNet161", 42 + fold["fold"], fold["train"], fold["test"], cache, trials,
                      args.epochs, 16, args.threads, results / "runs" / f"cv_DenseNet161_fold{fold['fold']}", False)
    print("Requested experiments complete", flush=True)


if __name__ == "__main__":
    main()
