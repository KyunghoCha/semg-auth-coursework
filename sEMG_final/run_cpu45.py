"""Run the frozen 45-epoch comparison on two isolated four-core CPU slots.

Holdout training never evaluates the final test. A one-epoch pilot uses the
same configuration and can be resumed with the command without --pilot-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    out = Path(args.output_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    args.cache_dir = str(Path(args.cache_dir).resolve())
    guard = (out / '.orchestrator.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 4 * args.workers:
        raise RuntimeError('Insufficient CPU cores; use --workers 1 for one four-core slot')
    slots = [available[4*i:4*(i+1)] for i in range(args.workers)]
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    # Prepare once under a shared cache lock before concurrent readers start.
    with (cache_dir / '.prepare.lock').open('a') as cache_guard:
        fcntl.flock(cache_guard, fcntl.LOCK_EX)
        prepare = ('from data_utils import load_trials; from experiment import make_plan; '
                   'from colab_run import build_cache; from experiment import save_json,write_csv; from pathlib import Path; import sys; '
                   'trials,_=load_trials(sys.argv[1]); '
                   'plan=make_plan(trials); build_cache(trials,plan,Path(sys.argv[2]),"lecture"); '
                   'out=Path(sys.argv[3]); save_json(out/"splits.json",plan); '
                   'write_csv(out/"manifest.csv",[{"index":i,"path":t.path,"subject":t.subject,"trial":t.number,'
                   '"sha256":t.sha256,"signal_sha256":t.signal_sha256} for i,t in enumerate(trials)])')
        subprocess.run([sys.executable, '-c', prepare, str(ROOT / 'data/data'), str(cache_dir), str(out)],
                       cwd=ROOT, check=True)

    jobs = [{'name': f'holdout_DenseNet161_seed{s}', 'job': 'holdout', 'model': 'DenseNet161', 'seed': s}
            for s in [42, 43, 44]]
    jobs += [{'name': f'cv_DenseNet161_fold{f}', 'job': 'cv', 'model': 'DenseNet161', 'seed': 42+f, 'fold': f}
             for f in range(1, 6)]
    jobs += [{'name': f'holdout_{m}_seed{s}', 'job': 'holdout', 'model': m, 'seed': s}
             for m in ['ResNet18', 'SimpleCNN'] for s in [42, 43, 44]]
    if args.pilot_only:
        jobs = jobs[:2]
    pending = queue.Queue()
    for job in jobs: pending.put(job)
    mutex = threading.Lock(); stopped = threading.Event()
    status = {'started_at_utc': utc(), 'pilot_only': args.pilot_only, 'cpu_slots': slots,
              'jobs': {job['name']: {'status': 'queued'} for job in jobs}}
    progress_path = out / ('pilot_progress.json' if args.pilot_only else 'progress.json')

    def update(name, values):
        with mutex:
            status['jobs'][name] = {**status['jobs'][name], **values}
            status['updated_at_utc'] = utc()
            temporary = progress_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(progress_path)
            print(json.dumps({'job': name, **values}), flush=True)

    def worker(slot):
        while not stopped.is_set():
            try: job = pending.get_nowait()
            except queue.Empty: return
            path = out / 'runs' / job['name']; path.mkdir(parents=True, exist_ok=True)
            command = ['flock', '--nonblock', '--conflict-exit-code', '75', str(path / '.training.lock'),
                       'taskset', '-c', ','.join(map(str, slots[slot])), sys.executable, '-u',
                       str(ROOT / 'colab_run.py'), '--data-dir', str(ROOT / 'data/data'),
                       '--cache-dir', args.cache_dir, '--output-dir', str(path), '--job', job['job'],
                       '--model', job['model'], '--seed', str(job['seed']), '--epochs', '45',
                       '--device', 'cpu', '--threads', '4', '--checkpoint-every', '5']
            if 'fold' in job: command += ['--fold', str(job['fold'])]
            if args.pilot_only: command += ['--max-epochs-this-invocation', '1']
            started = time.perf_counter()
            update(job['name'], {'status': 'running', 'started_at_utc': utc(), 'cpu_affinity': slots[slot]})
            with (path / 'execution.jsonl').open('a') as stream:
                stream.write(json.dumps({'at_utc': utc(), 'command': command, 'cpu_affinity': slots[slot],
                                         'concurrent_slots': args.workers, 'pilot': args.pilot_only})+'\n')
            env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            with (path / 'training.log').open('a') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
            elapsed = time.perf_counter() - started
            if result.returncode:
                update(job['name'], {'status': 'blocked_existing_job' if result.returncode == 75 else 'failed',
                                     'returncode': result.returncode, 'wall_seconds': elapsed})
                stopped.set()
                return
            complete = (path / 'completion.json').exists()
            update(job['name'], {'status': 'trained' if complete else 'paused', 'returncode': 0,
                                 'finished_at_utc': utc(), 'wall_seconds': elapsed})
    def guarded_worker(slot):
        try:
            return worker(slot)
        except Exception as error:
            stopped.set()
            active = [name for name, row in status['jobs'].items()
                      if row.get('status') == 'running' and row.get('cpu_affinity') == slots[slot]]
            for name in active:
                update(name, {'status': 'failed', 'exception': repr(error)})
            raise
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(guarded_worker, slot) for slot in range(args.workers)]
        for future in futures: future.result()
    if stopped.is_set():
        raise SystemExit('A job needs attention; no additional jobs were started after the failure')
    print('Pilot paused with checkpoints' if args.pilot_only else 'All frozen training jobs complete', flush=True)


if __name__ == '__main__':
    main()
