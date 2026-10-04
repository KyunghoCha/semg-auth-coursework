"""Sequentially evaluate every frozen comparison variant after all training ends."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    if not args.evaluate_test:
        parser.error('Use --evaluate-test only after accepting the frozen, reused-holdout evaluation plan')
    root = Path(args.run_root).resolve()
    plan = root / 'frozen_plan.json'
    planned = json.loads(plan.read_text())
    plan_hash = sha(plan)
    jobs = [(f'holdout_{m}_seed{s}', m, 'holdout')
            for m in planned['models'] for s in planned['holdout_training_seeds']]
    jobs += [(f'cv_DenseNet161_fold{f}', 'DenseNet161', 'cv') for f in planned['cv_folds']]
    # Hold every training lock for the entire evaluation pass: no concurrent training writers.
    locks = []
    for name, _, _ in jobs:
        path = root / 'runs' / name
        completion = json.loads((path / 'completion.json').read_text())
        if completion['epochs'] != planned['epochs']:
            raise ValueError(f'Incomplete training: {name}')
        lock = (path / '.training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    eval_lock = (root / '.evaluation.lock').open('a')
    fcntl.flock(eval_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    for name, model, phase in jobs:
        source_config = json.loads((root/'runs'/name/'config.json').read_text())
        source_config = {k:v for k,v in source_config.items()
                         if k not in ['duplicates_removed','device_name','cuda_version','test_evaluation_requested']}
        for variant in (['base', 'bn'] if model == 'DenseNet161' else ['base']):
            target = root / 'evaluation' / name / variant
            if (target / 'metrics.json').exists():
                saved = json.loads((target / 'metrics.json').read_text())
                expected_model = model + ('+BN' if variant == 'bn' else '')
                if saved['model'] != expected_model or saved['bn_recalibration'] != (variant == 'bn') or saved['config'] != source_config or saved['test_evaluated'] != (phase == 'holdout'):
                    raise ValueError('Existing evaluation has the wrong variant or source configuration')
                if saved['plan_sha256'] != plan_hash or saved['evaluation_code_sha256'] != sha(ROOT / 'finalize45.py') or saved['source_checkpoint_sha256'] != sha(root/'runs'/name/'checkpoint.pt'):
                    raise ValueError('Existing evaluation has a different frozen plan or evaluator')
                print(f'Preserved completed evaluation: {name}/{variant}', flush=True)
                continue
            if target.exists():
                target.rename(target.with_name(target.name+'.incomplete-'+uuid.uuid4().hex[:8]))
            command = [sys.executable, '-u', str(ROOT / 'finalize45.py'), '--run-dir', str(root/'runs'/name),
                       '--output-dir', str(target), '--plan', str(plan), '--data-dir', str(ROOT/'data/data'),
                       '--cache-dir', str(Path(args.cache_dir).resolve())]
            if variant == 'bn': command += ['--bn-recalibrate']
            if phase == 'holdout': command += ['--evaluate-test']
            subprocess.run(command, cwd=ROOT, env=env, check=True,
                           pass_fds=tuple(lock.fileno() for lock in locks+[eval_lock]))
    if sha(plan) != plan_hash:
        raise ValueError('Frozen plan changed during evaluation batch')
    print('All 22 frozen evaluations complete; no variant was selected using holdout scores', flush=True)


if __name__ == '__main__':
    main()
