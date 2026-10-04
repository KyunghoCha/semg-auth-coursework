"""One prespecified, training-only BatchNorm recalibration diagnostic.

This command accepts completed internal-validation runs only. It cannot evaluate
or fit to the final holdout. Original checkpoints are never modified.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import update_bn

from colab_run import build_cache, evaluate
from data_utils import LABELS, load_trials
from experiment import TrialWindows, model_for, save_json, set_seed, write_csv


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_unchanged(before, after):
    if before.keys() != after.keys() or any(not torch.equal(before[k], after[k]) for k in before):
        raise AssertionError('Evaluation changed model state')


def prediction_rows(trials, indices, actual, predicted):
    return [{'path': trials[indices[i // 19]].path, 'window': i % 19,
             'true': LABELS[y], 'predicted': LABELS[p]}
            for i, (y, p) in enumerate(zip(actual, predicted))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-dir', default='data/data')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    run = Path(args.run_dir)
    out = Path(args.output_dir)
    if out.exists():
        raise ValueError('Use a new output directory; existing results are preserved')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_seed(2026)
    device = torch.device('cpu')
    stored = json.loads((run / 'metrics.json').read_text())
    config = stored['config']
    if config['job'] != 'validate' or config['epochs'] != 45:
        raise ValueError('Only completed 45-epoch internal-validation runs are allowed')
    plan = json.loads((run / 'splits.json').read_text())
    train_ids, val_ids = config['train_indices'], config['evaluation_indices']
    test_ids = plan['holdout']['test']
    for indices in [train_ids, val_ids, test_ids]:
        if len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 or i >= 249 for i in indices):
            raise AssertionError('Duplicate or invalid trial indices')
    if set(train_ids) | set(val_ids) != set(plan['holdout']['train']):
        raise AssertionError('Internal split differs from prescribed development set')
    for name, expected in config['code_sha256'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise AssertionError(f'Source code changed: {name}')
    if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & set(test_ids):
        raise AssertionError('Trial leakage')
    trials, _ = load_trials(args.data_dir)
    if [t.signal_sha256 for t in trials] != config['source_signal_hashes']:
        raise AssertionError('Data differs from source run')
    if (len(train_ids), len(val_ids), len(test_ids)) != (159, 40, 50):
        raise AssertionError('Unexpected prescribed split')
    checkpoint_path = run / 'checkpoint.pt'
    checkpoint_sha = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != 45 or checkpoint['config'] != config:
        raise AssertionError('Incomplete or incompatible checkpoint')
    original = model_for(config['model']).to(device)
    original.load_state_dict(checkpoint['model'])
    del checkpoint
    gc.collect()
    original.eval()
    cache = build_cache(trials, plan, Path(args.cache_dir), config['filter_mode'])
    train_data = TrialWindows(cache, trials, train_ids)
    val_data = TrialWindows(cache, trials, val_ids)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0)
    before = snapshot(original)
    original_scores, actual, original_pred = evaluate(original, val_loader, device)
    assert_unchanged(before, snapshot(original))
    repeated_scores, repeated_true, repeated_pred = evaluate(original, val_loader, device)
    assert_unchanged(before, snapshot(original))
    if (actual, original_pred, original_scores) != (repeated_true, repeated_pred, repeated_scores):
        raise AssertionError('Original evaluation was not deterministic')
    for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
        if not np.isclose(original_scores[metric], stored[metric], rtol=0, atol=1e-12):
            raise AssertionError(f'Original {metric} differs from saved run')
    if original_scores['confusion_matrix'] != stored['confusion_matrix']:
        raise AssertionError('Original confusion matrix differs')
    with (run / 'predictions.csv').open(newline='') as stream:
        saved_predictions = list(csv.DictReader(stream))
    reconstructed = [{**row, 'window': str(row['window'])}
                     for row in prediction_rows(trials, val_ids, actual, original_pred)]
    if reconstructed != saved_predictions:
        raise AssertionError('Original per-window predictions differ from source run')
    allowed_suffixes = ('.running_mean', '.running_var', '.num_batches_tracked')
    calibrated_states, calibrated_scores, calibrated_predictions = [], [], []
    calibration_seconds = []
    for repetition in range(2):
        candidate = copy.deepcopy(original)
        loader = DataLoader(train_data, batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(2026),
                            num_workers=0, drop_last=False)
        initial_momenta = {n: m.momentum for n, m in candidate.named_modules()
                           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
        start = perf_counter()
        update_bn(loader, candidate, device=device)
        calibration_seconds.append(perf_counter() - start)
        if candidate.training:
            raise AssertionError('Recalibration failed to restore evaluation mode')
        for name, module in candidate.named_modules():
            if name in initial_momenta and module.momentum != initial_momenta[name]:
                raise AssertionError('Recalibration changed BN momentum')
        candidate.eval()
        after = snapshot(candidate)
        for name, value in after.items():
            if name.endswith(allowed_suffixes):
                if not torch.isfinite(value).all():
                    raise AssertionError('Non-finite BN statistics')
                if name.endswith('.num_batches_tracked') and int(value) != len(loader):
                    raise AssertionError('Unexpected BN update count')
            elif not torch.equal(value, before[name]):
                raise AssertionError(f'Non-BN state changed: {name}')
        scores, truth, predicted = evaluate(candidate, val_loader, device)
        assert_unchanged(after, snapshot(candidate))
        if truth != actual:
            raise AssertionError('Validation order changed')
        calibrated_states.append(after)
        calibrated_scores.append(scores)
        calibrated_predictions.append(predicted)
        del candidate
    assert_unchanged(calibrated_states[0], calibrated_states[1])
    if calibrated_scores[0] != calibrated_scores[1] or calibrated_predictions[0] != calibrated_predictions[1]:
        raise AssertionError('Recalibration is not reproducible')
    if sha(checkpoint_path) != checkpoint_sha:
        raise AssertionError('Original checkpoint changed')
    adopted = (calibrated_scores[0]['accuracy'] > original_scores['accuracy'] + 1e-12
               and calibrated_scores[0]['f1_macro'] >= original_scores['f1_macro'] - 1e-12)
    out.mkdir(parents=True)
    torch.save(calibrated_states[0], out / 'calibrated_weights.pt')
    write_csv(out / 'original_predictions.csv', prediction_rows(trials, val_ids, actual, original_pred))
    write_csv(out / 'calibrated_predictions.csv', prediction_rows(trials, val_ids, actual, calibrated_predictions[0]))
    result = {'source_run': str(run), 'source_checkpoint_sha256': checkpoint_sha,
              'source_config': config, 'calibration_seed': 2026, 'calibration_batches': len(loader),
              'calibration_windows': len(train_data), 'calibration_seconds': calibration_seconds,
              'validation_windows': len(val_data),
              'validation_trials': len(val_ids), 'original': original_scores,
              'recalibrated': calibrated_scores[0], 'selection_rule':
              'strictly higher validation accuracy and no decrease in macro F1',
              'adopt_candidate': adopted, 'checks_passed': True,
              'test_evaluated': False, 'calibration_code_sha256': sha(__file__)}
    save_json(out / 'comparison.json', result)
    print(json.dumps({k: result[k] for k in ['original', 'recalibrated', 'adopt_candidate', 'checks_passed']}, indent=2))


if __name__ == '__main__':
    main()
