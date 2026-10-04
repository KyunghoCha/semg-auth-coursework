"""Recompute every frozen evaluation from predictions, then produce tables/figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data_utils import LABELS
from summarize import draw_confusion

MODELS = ['SimpleCNN', 'ResNet18', 'DenseNet161', 'DenseNet161+BN']
METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']


def read_run(path,manifest=None):
    metrics = json.loads((path/'metrics.json').read_text())
    with (path/'predictions.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metrics['n_evaluation_windows'] or len({(r['path'],r['window']) for r in rows}) != len(rows):
        raise ValueError(f'Prediction count/uniqueness error: {path}')
    true = [r['true'] for r in rows]; pred = [r['predicted'] for r in rows]
    if not set(true+pred) <= set(LABELS): raise ValueError('Unknown identity label')
    precision, recall, f1, _ = precision_recall_fscore_support(true,pred,labels=LABELS,average='macro',zero_division=0)
    computed = dict(zip(METRICS,[accuracy_score(true,pred),precision,recall,f1]))
    for key,value in computed.items():
        if not np.isclose(metrics[key],value,rtol=0,atol=1e-12):
            raise ValueError(f'Metric mismatch: {path}/{key}')
    cm = confusion_matrix(true,pred,labels=LABELS)
    if cm.tolist() != metrics['confusion_matrix']:
        raise ValueError(f'Confusion matrix mismatch: {path}')
    counts = {}
    for row in rows: counts.setdefault(row['path'],[]).append(int(row['window']))
    if len(counts) != metrics['n_evaluation_trials'] or any(sorted(w) != list(range(19)) for w in counts.values()):
        raise ValueError('Each evaluation trial must appear once with all19 windows')
    if manifest is not None:
        config=metrics['config']
        if [manifest[i]['signal_sha256'] for i in range(len(manifest))] != config['source_signal_hashes']:
            raise ValueError('Manifest hashes differ from the frozen source')
        expected=[(manifest[i]['path'],str(w),manifest[i]['subject'])
                  for i in config['evaluation_indices'] for w in range(19)]
        if [(r['path'],r['window'],r['true']) for r in rows] != expected:
            raise ValueError('Predictions do not match prescribed trial/window/label membership')
    return metrics,rows


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(root):
    root=Path(root);out=root/'summary';out.mkdir(exist_ok=True)
    with (root/'manifest.csv').open(newline='') as stream:
        manifest={int(r['index']):r for r in csv.DictReader(stream)}
    splits=json.loads((root/'splits.json').read_text())
    comparison=[];all_runs=[];errors=[];plan_hashes=set();figures=[]
    for model in MODELS:
        base=model.replace('+BN','');variant='bn' if model.endswith('+BN') else 'base'
        runs=[]
        for seed in [42,43,44]:
            path=root/'evaluation'/f'holdout_{base}_seed{seed}'/variant
            run,rows=read_run(path,manifest);runs.append(run);plan_hashes.add(run['plan_sha256'])
            if run['config']['train_indices'] != splits['holdout']['train'] or run['config']['evaluation_indices'] != splits['holdout']['test']:
                raise ValueError('Holdout split differs from the prescribed split')
            if run['bn_recalibration'] != (variant=='bn'):
                raise ValueError('Wrong holdout variant flag')
            if run['model'] != model or run['n_evaluation_windows'] != 950 or run['config']['epochs'] != 45 or run['config']['job'] != 'holdout' or run['config']['seed'] != seed or run['n_train_trials'] != 199:
                raise ValueError('Unexpected comparison model/count/budget')
            all_runs.append({'phase':'holdout','model':model,'seed_or_fold':seed,
                             **{k:run[k] for k in METRICS},'train_seconds':run['train_seconds'],
                             'calibration_seconds':run['calibration_seconds']})
            if seed==42:
                cm=np.array(run['confusion_matrix'])
                if not np.array_equal(cm.sum(axis=1),np.full(5,190)): raise ValueError('Unbalanced holdout')
                figures.append((cm,f'{model}, seed42',out/f'confusion_{model}.png'))
                off=cm.copy();np.fill_diagonal(off,0);maximum=int(off.max())
                pairs=[(LABELS[i],LABELS[j]) for i,j in zip(*np.where(off==maximum)) if i!=j] if maximum else []
                examples=[next(r for r in rows if r['true']==a and r['predicted']==b) for a,b in pairs]
                a,b=pairs[0] if pairs else ('A','A')
                errors.append({'model':model,'seed':42,'maximum_error_count':maximum,
                    'maximum_error_pairs':pairs,'examples':examples,
                    'reverse_error_count':int(cm[LABELS.index(b),LABELS.index(a)]) if pairs else 0,
                    'destination_predictions':int(cm[:,LABELS.index(b)].sum()) if pairs else 0,
                    'destination_support':190})
        row={'model':model,'parameters':runs[0]['parameters'],'n_correct_total':sum(int(np.trace(r['confusion_matrix'])) for r in runs)}
        for metric in METRICS:
            row[metric+'_mean']=float(np.mean([r[metric] for r in runs]))
            row[metric+'_sd']=float(np.std([r[metric] for r in runs],ddof=0))
        for key in ['train_seconds','calibration_seconds','inference_ms_per_window']:
            row[key+'_mean']=float(np.mean([r[key] for r in runs]))
        row['fit_seconds_mean']=row['train_seconds_mean']+row['calibration_seconds_mean']
        comparison.append(row)
    cvs={};cv_rows=[]
    for variant in ['base','bn']:
        rows=[]
        for fold in range(1,6):
            run,predictions=read_run(root/'evaluation'/f'cv_DenseNet161_fold{fold}'/variant,manifest)
            expected_split=splits['cv'][fold-1]
            if run['config']['train_indices'] != expected_split['train'] or run['config']['evaluation_indices'] != expected_split['test']:
                raise ValueError('CV split differs from prescribed split')
            if run['model'] != ('DenseNet161+BN' if variant=='bn' else 'DenseNet161') or run['bn_recalibration'] != (variant=='bn'):
                raise ValueError('Wrong CV model variant')
            if run['config']['job'] != 'cv' or run['config']['fold'] != fold or run['config']['seed'] != 42+fold or run['config']['epochs'] != 45:
                raise ValueError('Unexpected CV configuration')
            if variant=='base':
                with (root/'runs'/f'cv_DenseNet161_fold{fold}'/'predictions.csv').open(newline='') as stream:
                    if predictions != list(csv.DictReader(stream)):
                        raise ValueError('Re-evaluated CV differs from original predictions')
            plan_hashes.add(run['plan_sha256'])
            row={'variant':variant,'fold':fold,'n_validation_trials':run['n_evaluation_trials'],
                 **{k:run[k] for k in METRICS}}
            rows.append(row);cv_rows.append(row)
            all_runs.append({'phase':'cv','model':run['model'],'seed_or_fold':fold,
                **{k:run[k] for k in METRICS},'train_seconds':run['train_seconds'],
                'calibration_seconds':run['calibration_seconds']})
        cvs[variant]={'folds':rows,**{metric+suffix:float(fn([r[metric] for r in rows]))
                     for metric in METRICS for suffix,fn in [('_mean',np.mean),('_sd',np.std)]}}
    if plan_hashes != {hashlib.sha256((root/'frozen_plan.json').read_bytes()).hexdigest()} or len(all_runs)!=22: raise ValueError('Incomplete/mixed frozen evaluations')
    for cm,title,destination in figures: draw_confusion(cm,title,destination)
    # The course submission checklist uses this conventional filename.
    shutil.copyfile(out/'confusion_DenseNet161.png', out/'confusion.png')
    ranking=sorted(comparison,key=lambda r:(-r['n_correct_total'],r['fit_seconds_mean']))
    summary={'comparison':comparison,'best_model':ranking[0]['model'],'worst_model':ranking[-1]['model'],
             'ranking_rule':'mean Accuracy over3seeds; tie by mean training+calibration time',
             'cv_DenseNet161':cvs,'errors_seed42':errors,'verified_evaluations':22,
             'standard_deviation':'population SD (ddof=0)','metric_unit':'300ms window',
             'holdout_status':'reused50trial holdout; five-epoch test scores previously observed',
             'plan_sha256':next(iter(plan_hashes))}
    write_csv(out/'comparison.csv',comparison);write_csv(out/'cross_validation.csv',cv_rows)
    write_csv(out/'all_evaluations.csv',all_runs)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    print(json.dumps(summarize(parser.parse_args().run_root),ensure_ascii=False,indent=2))
