"""GPU-capable 45-epoch runner with resumable state and a locked-test gate.

Select recipes on `validate` only. `holdout` evaluates the already-observed
50-trial test set only when --evaluate-test is explicitly supplied.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/semg-matplotlib")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from data_utils import DATA_COMMIT, LABELS, load_trials
from experiment import (TrialWindows, make_plan, make_windows, normalize_windows,
                        to_cwt, preprocess, model_for, set_seed, save_json, write_csv)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_state(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy_name": state[0],
            "numpy_keys": torch.tensor(state[1].astype(np.int64)),
            "numpy_pos": state[2], "numpy_gauss": state[3], "numpy_cached": state[4],
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"], state["numpy_keys"].numpy().astype(np.uint32),
                         state["numpy_pos"], state["numpy_gauss"], state["numpy_cached"]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, generator, config, epoch, history, train_seconds):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "rng": rng_state(generator), "config": config, "epoch": epoch,
             "history": history, "train_seconds": train_seconds}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, generator, config, device):
    # Only primitive metadata and tensors are serialized; no arbitrary pickle objects.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != config:
        raise ValueError("Checkpoint configuration differs. Use a different output folder for a new recipe.")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    for item in optimizer.state.values():
        for key, value in item.items():
            if torch.is_tensor(value) and key != "step":
                item[key] = value.to(device)
    restore_rng(state["rng"], generator)
    return state["epoch"], state["history"], state["train_seconds"]


def build_cache(trials, plan, directory, filter_mode):
    directory.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({"hashes": [trial.signal_sha256 for trial in trials],
        "filter_mode": filter_mode, "cwt": "morl_scales1to32_abs_thirdmean_window300_hop150_jointminmax"}, sort_keys=True).encode()).hexdigest()
    path = directory / "cwt.npy"; meta_path = directory / "metadata.json"
    if path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text())["signature"] != signature:
            raise ValueError("Cache recipe differs; choose a different --cache-dir")
        return np.load(path, mmap_mode="r")
    # The trial splits are fixed before overlapping windows are generated.
    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(trials), 19, 3, 32, 300))
    for position, index in enumerate(plan["holdout"]["train"] + plan["holdout"]["test"], 1):
        signal = trials[index].signal
        if filter_mode == "lecture":
            signal = preprocess(signal)
        windows = normalize_windows(make_windows(signal))
        cache[index] = np.stack([to_cwt(window) for window in windows])
        if position % 50 == 0:
            print(f"CWT {position}/{len(trials)}", flush=True)
    cache.flush(); save_json(meta_path, {"signature": signature, "filter_mode": filter_mode})
    del cache
    return np.load(path, mmap_mode="r")


def evaluate(model, loader, device):
    model.eval(); actual, predicted = [], []
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features.to(device, non_blocking=True))
            actual.extend(labels.tolist()); predicted.extend(logits.argmax(1).cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, labels=range(5), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(actual, predicted)), "precision_macro": float(precision),
            "recall_macro": float(recall), "f1_macro": float(f1),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=range(5)).tolist()}, actual, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/data")
    parser.add_argument("--cache-dir", default="cache_gpu/lecture")
    parser.add_argument("--output-dir", required=True, help="Persistent, unique directory for this one job")
    parser.add_argument("--job", choices=["validate", "holdout", "cv"], default="validate")
    parser.add_argument("--model", choices=["SimpleCNN", "ResNet18", "DenseNet161"], default="DenseNet161")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--filter-mode", choices=["lecture", "provided"], default="lecture")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-epochs-this-invocation", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.checkpoint_every < 1 or args.batch_size < 2:
        parser.error("Positive epochs/checkpoint interval and batch size >=2 required")
    if args.evaluate_test and args.job != "holdout":
        parser.error("--evaluate-test is only for a frozen holdout run")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No GPU is allocated. Select a GPU runtime; no CPU fallback or paid runtime is started.")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    trials, duplicates = load_trials(args.data_dir)
    plan = make_plan(trials)
    outer_train = plan["holdout"]["train"]
    if args.job == "validate":
        train_indices, evaluation_indices = train_test_split(outer_train, test_size=.2, random_state=123,
            stratify=[trials[i].subject for i in outer_train])
        role = "internal validation; final 50 test trials excluded"
    elif args.job == "cv":
        fold = plan["cv"][args.fold - 1]
        train_indices, evaluation_indices = fold["train"], fold["test"]
        role = "cross-validation within 199 development trials"
    else:
        train_indices, evaluation_indices = outer_train, plan["holdout"]["test"]
        role = "reused 50-trial holdout; 5-epoch baseline test scores previously observed"
    if set(train_indices) & set(evaluation_indices):
        raise ValueError("Trial leakage")
    if args.job != "holdout" and (set(train_indices) | set(evaluation_indices)) & set(plan["holdout"]["test"]):
        raise ValueError("Protected test trials entered development")
    config = {"job": args.job, "model": args.model, "seed": args.seed, "fold": args.fold if args.job == "cv" else None,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "optimizer": "Adam", "pretrained": False, "filter_mode": args.filter_mode, "device": args.device,
        "threads": args.threads, "data_commit": DATA_COMMIT, "source_signal_hashes": [t.signal_sha256 for t in trials],
        "train_indices": list(train_indices), "evaluation_indices": list(evaluation_indices), "evaluation_role": role,
        "torch_version": str(torch.__version__),
        "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ["colab_run.py", "experiment.py", "data_utils.py"]}}
    if (output / "metrics.json").exists():
        previous = json.loads((output / "metrics.json").read_text())
        if previous["config"] != config:
            raise ValueError("Completed result uses a different recipe. Choose another output folder.")
        print("This job is already complete; its stored evaluation was not repeated.", flush=True)
        return
    if (output / "config.json").exists():
        previous_config = json.loads((output / "config.json").read_text())
        if any(previous_config.get(key) != value for key, value in config.items()):
            raise ValueError("Existing output uses a different configuration. Choose another output folder.")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    save_json(output / "splits.json", plan)
    cache = build_cache(trials, plan, Path(args.cache_dir), args.filter_mode)
    train_data = TrialWindows(cache, trials, train_indices)
    evaluation_data = TrialWindows(cache, trials, evaluation_indices)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=device.type == "cuda")
    evaluation_loader = DataLoader(evaluation_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0, pin_memory=device.type == "cuda")
    model = model_for(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    checkpoint = output / "checkpoint.pt"
    completed_epoch, history, train_seconds = 0, [], 0.
    if checkpoint.exists():
        completed_epoch, history, train_seconds = load_checkpoint(checkpoint, model, optimizer, generator, config, device)
        print(f"Resuming completed epoch {completed_epoch}/{args.epochs}", flush=True)
    save_json(output / "config.json", {**config, "duplicates_removed": duplicates,
        "device_name": device_name, "cuda_version": torch.version.cuda,
        "test_evaluation_requested": args.evaluate_test})
    last_epoch = min(args.epochs, completed_epoch + args.max_epochs_this_invocation) if args.max_epochs_this_invocation else args.epochs
    for epoch in range(completed_epoch + 1, last_epoch + 1):
        model.train(); correct, total_loss = 0, 0.
        sync(device); start = time.perf_counter()
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features); loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(1) == labels).sum())
        sync(device); elapsed = time.perf_counter() - start; train_seconds += elapsed
        row = {"epoch": epoch, "train_loss": total_loss / len(train_data),
               "online_train_accuracy": correct / len(train_data), "train_seconds": elapsed,
               "device_name": device_name}
        if args.job != "holdout":
            scores, _, _ = evaluate(model, evaluation_loader, device)
            row.update({"validation_" + key: scores[key] for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]})
        history.append(row); write_csv(output / "history.csv", history)
        if epoch % args.checkpoint_every == 0 or epoch == last_epoch:
            save_checkpoint(checkpoint, model, optimizer, generator, config, epoch, history, train_seconds)
        print(json.dumps(row), flush=True)
    if last_epoch < args.epochs:
        print(f"Paused at epoch {last_epoch}; rerun the identical command to resume.", flush=True)
        return
    torch.save(model.state_dict(), output / "weights.pt")
    if args.job == "holdout" and not args.evaluate_test:
        save_json(output / "completion.json", {"epochs": args.epochs, "test_evaluated": False})
        print("Training complete. Test remains unevaluated; use --evaluate-test after locking the recipe.", flush=True)
        return
    scores, actual, predictions = evaluate(model, evaluation_loader, device)
    prediction_rows = []
    for offset, (truth, prediction) in enumerate(zip(actual, predictions)):
        trial = trials[evaluation_indices[offset // 19]]
        prediction_rows.append({"path": trial.path, "window": offset % 19,
                                "true": LABELS[truth], "predicted": LABELS[prediction]})
    write_csv(output / "predictions.csv", prediction_rows)
    sample = evaluation_data[0][0].unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(10): model(sample)
        sync(device); start = time.perf_counter()
        for _ in range(100): model(sample)
        sync(device); inference_ms = (time.perf_counter() - start) * 10
    result = {**scores, "config": config, "evaluation_role": role, "metric_unit": "300ms window",
        "n_train_trials": len(train_indices), "n_evaluation_trials": len(evaluation_indices),
        "n_train_windows": len(train_data), "n_evaluation_windows": len(evaluation_data),
        "train_seconds": train_seconds, "parameters": sum(p.numel() for p in model.parameters()),
        "inference_ms_per_window": inference_ms}
    save_json(output / "metrics.json", result)
    save_json(output / "completion.json", {"epochs": args.epochs, "test_evaluated": args.job == "holdout",
                                          "evaluation_role": role})
    print(json.dumps({key: result[key] for key in ["accuracy", "f1_macro", "evaluation_role"]}), flush=True)


if __name__ == "__main__":
    main()
