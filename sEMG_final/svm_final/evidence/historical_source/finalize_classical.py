"""Restore unchanged classical finalist development CV/full199 fit; no test."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss
import run_spectral as base
import calibrate_svm as cal
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"calibration/finalist"

def freeze():
    cal.verify_freeze();OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/"frozen_classical_plan.json").exists()
    _,split,_,_=base.metadata()
    value={"frozen_utc":base.utc(),"recipe":cal.RECIPE,"calibration":"sigmoid,ensemble=False,3trial-grouped training-only folds,seed2026",
           "outer_development_folds":split["cv"],"full_development_indices":split["holdout"]["train"],
           "cv_interpretation":"Postselection development stability, not nested recipe selection or independent generalization",
           "base_plan_sha256":base.digest(ROOT/"predeclared_plan.json"),"calibration_plan_sha256":base.digest(ROOT/"calibration/fixed_calibration_plan.json"),
           "restoration":"Same previously frozen classical finalist, no new selection","test_evaluation_allowed":False}
    base.write_json(OUT/"frozen_classical_plan.json",value);print("CLASSICAL_FINALIST_FROZEN",base.utc(),flush=True)

def verify():
    cal.verify_freeze();plan=json.loads((OUT/"frozen_classical_plan.json").read_text())
    assert plan["base_plan_sha256"]==base.digest(ROOT/"predeclared_plan.json")
    assert plan["calibration_plan_sha256"]==base.digest(ROOT/"calibration/fixed_calibration_plan.json")
    return plan

def run():
    plan=verify();assert not (OUT/"completion.json").exists();started=time.perf_counter()
    results=[];all_y=[];all_p=[];all_groups=[]
    for fold in range(1,6):
        folder=OUT/f"development_fold{fold}";assert not folder.exists()
        model,p,fit=cal.fit_development_fold(fold,folder)
        ds=cal.development_features(plan["outer_development_folds"][fold-1]["test"])
        met=base.measures(ds["y"],p.argmax(1));met["log_loss"]=float(log_loss(ds["y"],p,labels=range(5)))
        trial=base.measures(ds["y"].reshape(-1,19)[:,0],p.reshape(-1,19,5).mean(1).argmax(1))
        row={"fold":fold,"window":met,"trial_mean_probability":trial,"fit":fit};results.append(row);base.write_json(folder/"metrics.json",row)
        all_y.append(ds["y"]);all_p.append(p);all_groups.append(ds["groups"])
        print("DEVELOPMENT_FOLD",fold,met,flush=True)
    y,p,groups=np.concatenate(all_y),np.concatenate(all_p),np.concatenate(all_groups)
    assert len(y)==3781 and len(set(groups))==199 and set(groups)==set(plan["full_development_indices"])
    np.savez_compressed(OUT/"development_oof_probabilities.npz",y=y,p=p,groups=groups)
    stats={}
    for metric in ["accuracy","f1_macro","log_loss"]:
        values=np.array([row["window"][metric] for row in results])
        stats[metric]={"fold_values":values.tolist(),"equal_fold_mean":float(values.mean()),"population_sd_ddof0":float(values.std(ddof=0)),"sample_sd_ddof1":float(values.std(ddof=1))}
    pooled=base.measures(y,p.argmax(1));pooled["log_loss"]=float(log_loss(y,p,labels=range(5)))
    directory=OUT/"full_development";assert not directory.exists();model,fit=cal.fit_full_development(directory);path=directory/"model.joblib"
    result={"completed_utc":base.utc(),"plan_sha256":base.digest(OUT/"frozen_classical_plan.json"),"folds":results,"cv_summary":stats,"pooled_oof_window":pooled,
            "cv_interpretation":plan["cv_interpretation"],"full_development_fit":fit,"full_model_sha256":base.digest(path),"full_model_bytes":path.stat().st_size,
            "total_seconds":time.perf_counter()-started,"test_evaluated":False,"test_signal_files_opened":0}
    base.write_json(OUT/"completion.json",result);print("CLASSICAL_FINALIST_COMPLETE",stats["accuracy"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["freeze","run"])
    {"freeze":freeze,"run":run}[parser.parse_args().stage]()
