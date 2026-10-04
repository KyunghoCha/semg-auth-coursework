"""Read complete sEMG trials and split without shared duplicate signals."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import train_test_split


DATA_COMMIT = "adb7955f4416165c88e4111af6f8fdafd416209c"
LABELS = list("ABCDE")


@dataclass
class Trial:
    path: str
    subject: str
    number: int
    sha256: str
    signal_sha256: str
    signal: np.ndarray


def load_trials(data_dir):
    """Validate all 250 source files; retain one copy of each identical signal."""
    root = Path(data_dir)
    files = sorted(root.glob("*/*.csv"), key=lambda p: (p.parent.name, trial_number(p)))
    if len(files) != 250:
        raise ValueError(f"Expected 250 source CSV files in {root}, found {len(files)}")
    trials, duplicates, seen, numbers = [], [], {}, {label: set() for label in LABELS}
    for path in files:
        subject = path.parent.name
        if subject not in LABELS:
            raise ValueError(f"Unknown subject: {path}")
        number = trial_number(path)
        if number in numbers[subject]:
            raise ValueError(f"Duplicate subject/trial key: {path}")
        numbers[subject].add(number)
        content = path.read_bytes()
        if content.decode("utf-8-sig").splitlines()[0].strip() != "Comp Ch 3,Comp Ch 4":
            raise ValueError(f"Unexpected CSV header: {path}")
        signal = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if signal.shape != (3000, 2) or not np.isfinite(signal).all():
            raise ValueError(f"Invalid shape or values: {path}")
        digest = hashlib.sha256(signal.astype("<f8").tobytes()).hexdigest()
        relative = path.relative_to(root).as_posix()
        if digest in seen:
            kept = seen[digest]
            if kept.subject != subject:
                raise ValueError(f"Identical signals have conflicting labels: {relative}")
            duplicates.append({"removed": relative, "kept": kept.path})
            continue
        trial = Trial(relative, subject, number, hashlib.sha256(content).hexdigest(), digest, signal)
        seen[digest] = trial
        trials.append(trial)
    if any(values != set(range(1, 51)) for values in numbers.values()):
        raise ValueError("Each subject must have trial numbers 1 through 50")
    return trials, duplicates


def trial_number(path):
    match = re.search(r"\((\d+)\)", Path(path).stem)
    if match is None:
        raise ValueError(f"Missing trial number: {path}")
    return int(match.group(1))


def split_trials(trials, test_size, seed):
    """Split whole trials, stratified by registered identity, before preprocessing."""
    indices = np.arange(len(trials))
    labels = np.array([trial.subject for trial in trials])
    train, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
    train_hashes = {trials[i].signal_sha256 for i in train}
    test_hashes = {trials[i].signal_sha256 for i in test}
    if train_hashes & test_hashes:
        raise ValueError("Identical signals occur in both train and test")
    if set(labels[train]) != set(LABELS) or set(labels[test]) != set(LABELS):
        raise ValueError("All registered users must occur in train and test")
    return train, test
