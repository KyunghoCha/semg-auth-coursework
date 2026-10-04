"""Evaluate frozen epoch-45 models, optionally applying the selected DenseNet BN variant.

Run after training finishes. --evaluate-test explicitly permits the reused
50-trial holdout; otherwise only development-fold evaluation is allowed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile

import torch
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader

from colab_run import build_cache, evaluate
from data_utils import LABELS, load_trials
from experiment import TrialWindows, make_plan, model_for, save_json, set_seed, write_csv


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data-dir', default='data/data')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--bn-recalibrate', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    run = Path(args.run_dir); out = Path(args.output_dir)
    if out.exists():
        raise ValueError('Use a new output folder; existing evaluations are preserved')
    plan_hash = sha(args.plan)
    frozen = json.loads(Path(args.plan).read_text())
    if frozen['epochs'] != 45 or frozen['checkpoint_rule'] != 'fixed final epoch45; no selection by final-test scores':
        raise ValueError('Unexpected frozen plan')
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 4:
        raise RuntimeError('Four CPU cores are required for this frozen benchmark')
    os.sched_setaffinity(0, available[:4])
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    set_seed(2026)
    device = torch.device('cpu')
    completion = json.loads((run / 'completion.json').read_text())
    if completion['epochs'] != 45:
        raise ValueError('Training is incomplete')
    checkpoint_hash = sha(run / 'checkpoint.pt')
    state = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=True)
    config = state['config']
    saved_config = json.loads((run / 'config.json').read_text())
    if state['epoch'] != 45 or config['epochs'] != 45 or any(saved_config.get(k) != v for k, v in config.items()):
        raise ValueError('Checkpoint/config mismatch')
    for key, plan_key in [('optimizer','optimizer'), ('pretrained','pretrained'), ('data_commit','dataset_commit'),
                          ('batch_size','batch_size'), ('learning_rate','learning_rate'), ('filter_mode','filter_mode')]:
        if config[key] != frozen[plan_key]:
            raise ValueError(f'Frozen recipe mismatch: {key}')
    for key in ['device', 'threads', 'torch_version']:
        if config[key] != frozen['runtime'][key]:
            raise ValueError(f'Frozen runtime mismatch: {key}')
    if str(torch.__version__) != frozen['runtime']['torch_version']:
        raise ValueError('Evaluator PyTorch version differs from the frozen runtime')
    if config['job'] == 'holdout':
        if config['model'] not in frozen['models'] or config['seed'] not in frozen['holdout_training_seeds'] or config['fold'] is not None:
            raise ValueError('Unplanned holdout model/seed/fold')
    elif config['job'] == 'cv':
        if config['model'] != frozen['cv_model'] or config['fold'] not in frozen['cv_folds']:
            raise ValueError('Unplanned CV model/fold')
        position = frozen['cv_folds'].index(config['fold'])
        if config['seed'] != frozen['cv_seeds'][position]:
            raise ValueError('Unplanned CV seed')
    if args.bn_recalibrate and frozen['bn_recalibration'] != {'model':'DenseNet161','seed':2026,'batch_size':16,'passes':1,'training_only':True}:
        raise ValueError('BN procedure differs from the selected recipe')
    if config['job'] not in ['holdout', 'cv']:
        raise ValueError('This evaluator accepts final-comparison jobs only')
    if config['job'] == 'holdout' and not args.evaluate_test:
        raise ValueError('Explicit --evaluate-test is required for the frozen holdout')
    if args.bn_recalibrate and config['model'] != 'DenseNet161':
        raise ValueError('The selected additional BN variant is DenseNet161 only')
    for name, expected in config['code_sha256'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError(f'Frozen source code changed: {name}')
    if config['batch_size'] != 16 or config['learning_rate'] != .001 or config['filter_mode'] != 'lecture':
        raise ValueError('Training recipe differs from the frozen plan')
    trials, _ = load_trials(args.data_dir)
    if [t.signal_sha256 for t in trials] != config['source_signal_hashes']:
        raise ValueError('Source dataset changed')
    plan = make_plan(trials)
    if plan != json.loads((run / 'splits.json').read_text()):
        raise ValueError('Prescribed split changed')
    expected_split = plan['holdout'] if config['job'] == 'holdout' else plan['cv'][config['fold']-1]
    train_ids, evaluation_ids = config['train_indices'], config['evaluation_indices']
    if train_ids != expected_split['train'] or evaluation_ids != expected_split['test']:
        raise ValueError('Run does not use its prescribed split')
    if set(train_ids) & set(evaluation_ids):
        raise ValueError('Trial leakage')
    if config['job'] == 'cv' and (set(train_ids) | set(evaluation_ids)) & set(plan['holdout']['test']):
        raise ValueError('Final test entered cross-validation')
    model = model_for(config['model']).to(device)
    model.load_state_dict(state['model'])
    train_seconds = state['train_seconds']
    del state
    model.eval()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    cache = build_cache(trials, plan, Path(args.cache_dir), 'lecture')
    training_data = TrialWindows(cache, trials, train_ids)
    evaluation_data = TrialWindows(cache, trials, evaluation_ids)
    calibration_seconds = 0.
    if args.bn_recalibrate:
        loader = DataLoader(training_data, batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(2026), num_workers=0, drop_last=False)
        start = time.perf_counter()
        update_bn(loader, model, device=device)
        calibration_seconds = time.perf_counter() - start
        model.eval()
        for name, value in model.state_dict().items():
            if name.endswith(('.running_mean', '.running_var', '.num_batches_tracked')):
                if not torch.isfinite(value).all():
                    raise AssertionError('Non-finite BN state')
                if name.endswith('.num_batches_tracked') and int(value) != len(loader):
                    raise AssertionError('Unexpected calibration count')
            elif not torch.equal(value, original_state[name]):
                raise AssertionError(f'Calibration changed learned state: {name}')
    before_evaluation = {k: v.clone() for k, v in model.state_dict().items()}
    evaluation_loader = DataLoader(evaluation_data, batch_size=16, shuffle=False, num_workers=0)
    scores, actual, predictions = evaluate(model, evaluation_loader, device)
    if any(not torch.equal(v, before_evaluation[k]) for k, v in model.state_dict().items()):
        raise AssertionError('Evaluation changed model state')
    rows = [{'path': trials[evaluation_ids[i // 19]].path, 'window': i % 19,
             'true': LABELS[y], 'predicted': LABELS[p]}
            for i, (y, p) in enumerate(zip(actual, predictions))]
    sample = evaluation_data[0][0].unsqueeze(0)
    with torch.inference_mode():
        for _ in range(10): model(sample)
        start = time.perf_counter()
        for _ in range(100): model(sample)
        inference_ms = (time.perf_counter() - start) * 10
    if sha(run / 'checkpoint.pt') != checkpoint_hash:
        raise AssertionError('Original checkpoint changed')
    if sha(args.plan) != plan_hash:
        raise AssertionError('Frozen plan changed during evaluation')
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name+'.tmp-', dir=out.parent))
    write_csv(staging / 'predictions.csv', rows)
    variant = config['model'] + ('+BN' if args.bn_recalibrate else '')
    result = {**scores, 'model': variant, 'config': config, 'metric_unit': '300ms window',
              'n_train_trials': len(train_ids), 'n_evaluation_trials': len(evaluation_ids),
              'n_train_windows': len(training_data), 'n_evaluation_windows': len(evaluation_data),
              'train_seconds': train_seconds, 'calibration_seconds': calibration_seconds,
              'parameters': sum(p.numel() for p in model.parameters()),
              'inference_ms_per_window': inference_ms, 'source_checkpoint_sha256': checkpoint_hash,
              'plan_sha256': plan_hash, 'evaluation_cpu_affinity': sorted(os.sched_getaffinity(0)), 'evaluation_code_sha256': sha(__file__),
              'evaluation_role': config['evaluation_role'], 'test_evaluated': config['job'] == 'holdout',
              'bn_recalibration': args.bn_recalibrate, 'calibration_seed': 2026 if args.bn_recalibrate else None,
              'learned_parameters_unchanged': True}
    save_json(staging / 'metrics.json', result)
    staging.rename(out)
    print(json.dumps({'model': variant, 'accuracy': scores['accuracy'], 'f1_macro': scores['f1_macro'],
                      'evaluation_role': config['evaluation_role']}), flush=True)


if __name__ == '__main__':
    main()
