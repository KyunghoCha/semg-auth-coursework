"""Restore the single fixed grouped sigmoid calibration, not a new search."""
import argparse
import csv
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import run_spectral as base

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"calibration"
RECIPE={"family":"combined","model":"rbf_svm","C":1.0}

def grouped_calibration_folds(y,groups):
    ordered=list(dict.fromkeys(int(g) for g in groups));labels=[]
    for group in ordered:
        ys=np.unique(y[groups==group]);assert len(ys)==1 and np.sum(groups==group)==19
        labels.append(int(ys[0]))
    folds,provenance=[],[];seen=np.zeros(len(y),int)
    for number,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(ordered,labels),1):
        tg=[ordered[j] for j in tr];vg=[ordered[j] for j in va]
        ti=np.flatnonzero(np.isin(groups,tg));vi=np.flatnonzero(np.isin(groups,vg))
        assert not set(tg)&set(vg) and len(ti)+len(vi)==len(y)
        assert set(y[ti])==set(range(5)) and set(y[vi])==set(range(5))
        seen[vi]+=1;folds.append((ti,vi))
        provenance.append({"fold":number,"fit_trial_indices":tg,"calibration_trial_indices":vg,"fit_window_count":len(ti),"calibration_window_count":len(vi)})
    assert np.all(seen==1)
    return folds,provenance

def assert_recipe_model(model,x):
    assert model.ensemble is False and model.method=="sigmoid" and model.n_jobs==1
    assert len(model.calibrated_classifiers_)==1 and np.array_equal(model.classes_,np.arange(5))
    pipe=model.calibrated_classifiers_[0].estimator
    assert pipe[1].C==1. and pipe[1].kernel=="rbf" and pipe[1].gamma=="scale" and pipe[1].probability is False
    assert int(pipe[0].n_samples_seen_)==len(x)
    assert np.allclose(pipe[0].mean_,x.mean(0),rtol=0,atol=1e-12)
    assert np.allclose(pipe[0].var_,x.var(0),rtol=0,atol=1e-12)
    return pipe

def fit_fixed_calibrated(x,y,groups):
    plan=base.get_plan()
    assert x.ndim==2 and x.shape[1]==39 and len(x)==len(y)==len(groups) and np.isfinite(x).all()
    assert set(groups).issubset(set(plan["train_indices"]+plan["validation_indices"]))
    folds,provenance=grouped_calibration_folds(y,groups)
    model=CalibratedClassifierCV(estimator=base.estimator(RECIPE),method="sigmoid",cv=folds,n_jobs=1,ensemble=False)
    started=time.perf_counter();model.fit(x,y);seconds=time.perf_counter()-started
    pipe=assert_recipe_model(model,x)
    return model,{"calibration_folds":provenance,"fit_seconds":seconds,"training_trials":len(set(groups)),"training_windows":len(y),
                  "full_fit_gamma":float(pipe[1]._gamma),"full_fit_support_vectors":int(pipe[1].n_support_.sum()),"class_order":base.LABELS,
                  "algorithm":"3 grouped OOF score fits plus one full-training base refit; sigmoid,ensemble=False","train_only_scaler_verified":True}

def predict_probabilities(model,x):
    assert np.array_equal(model.classes_,np.arange(5))
    p=model.predict_proba(x)
    assert p.shape==(len(x),5) and np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(1),1,atol=1e-12,rtol=0)
    return p

def write_predictions(path,ds,p):
    base.write_predictions(path,ds,p.argmax(1),p)

def freeze():
    plan=base.get_plan();OUT.mkdir(exist_ok=True)
    assert not (OUT/"fixed_calibration_plan.json").exists()
    selected=json.loads((ROOT/"selected_recipes.json").read_text())["selected"]["combined"]["recipe"]
    assert selected==RECIPE
    ds=np.load(ROOT/"train_features.npz");_,folds=grouped_calibration_folds(ds["y"],ds["groups"])
    for actual,expected in zip(folds,plan["inner_folds"]):
        assert actual["fit_trial_indices"]==expected["train_trial_indices"]
        assert actual["calibration_trial_indices"]==expected["validation_trial_indices"]
    value={"frozen_utc":base.utc(),"restoration":"Previously approved fixed single postdiagnostic variant; no new calibration selection",
           "recipe":RECIPE,"method":"sigmoid","ensemble":False,"n_jobs":1,"folds":folds,
           "selection_breadth":"Original18recipe grid plus one previously approved fixed calibration; no further feature/C/gamma/weight search",
           "base_plan_sha256":base.digest(ROOT/"predeclared_plan.json"),
           "artifacts_sha256":{name:base.digest(ROOT/name) for name in ["run_spectral.py","calibrate_svm.py","train_features.npz","validation_features.npz","selected_recipes.json"]},
           "test_evaluation":False}
    base.write_json(OUT/"fixed_calibration_plan.json",value);print("CALIBRATION_FROZEN",base.utc(),flush=True)

def verify_freeze():
    base.get_plan();plan=json.loads((OUT/"fixed_calibration_plan.json").read_text())
    assert base.digest(ROOT/"predeclared_plan.json")==plan["base_plan_sha256"]
    for name,sha in plan["artifacts_sha256"].items():assert base.digest(ROOT/name)==sha,name
    return plan

def run_validation():
    plan=verify_freeze();assert not (OUT/"validation_results.json").exists()
    tr=np.load(ROOT/"train_features.npz");model,fit=fit_fixed_calibrated(tr["X"],tr["y"],tr["groups"])
    assert fit["calibration_folds"]==plan["folds"]
    path=OUT/"model_calibrated_combined.joblib";joblib.dump(model,path)
    base.write_json(OUT/"fit_provenance.json",{"fit_completed_utc":base.utc(),"fit":fit,"model_sha256":base.digest(path),"plan_sha256":base.digest(OUT/"fixed_calibration_plan.json")})
    ds=np.load(ROOT/"validation_features.npz");p=predict_probabilities(model,ds["X"]);pred=p.argmax(1)
    met=base.measures(ds["y"],pred);met["log_loss"]=float(log_loss(ds["y"],p,labels=range(5)))
    met["brier_multiclass_sum"]=float(np.mean(np.sum((p-np.eye(5)[ds["y"]])**2,axis=1)))
    trial_mean=base.measures(ds["y"].reshape(-1,19)[:,0],p.reshape(-1,19,5).mean(1).argmax(1))
    trial_vote=base.measures(*base.trial_vote(ds["y"],pred));write_predictions(OUT/"validation_probabilities.csv",ds,p)
    np.savez_compressed(OUT/"validation_probabilities.npz",p=p,y=ds["y"],groups=ds["groups"],windows=ds["windows"])
    old=list(csv.DictReader((ROOT/"validation_predictions_combined.csv").open()))
    old_pred=np.array([base.LABELS.index(row["predicted"]) for row in old])
    assert np.array_equal(ds["groups"],[int(row["trial_index"]) for row in old])
    correct=pred==ds["y"];old_correct=old_pred==ds["y"];sample=base.bootstrap_samples(ds["y"])
    boot=correct.reshape(-1,19).mean(1)[sample].mean(1)
    diff=(correct.astype(float)-old_correct).reshape(-1,19).mean(1)[sample].mean(1)
    result={"completed_utc":base.utc(),"fit":fit,"window":met,"trial_mean_probability":trial_mean,"trial_majority_vote":trial_vote,
            "descriptive_window_accuracy_ci95":np.quantile(boot,[.025,.975]).tolist(),
            "change_from_original":{"window_accuracy_delta":float(correct.mean()-old_correct.mean()),"newly_correct":int(np.sum(correct&~old_correct)),
                                    "newly_wrong":int(np.sum(~correct&old_correct)),"changed_predicted_class":int(np.sum(pred!=old_pred)),"paired_descriptive_ci95":np.quantile(diff,[.025,.975]).tolist()},
            "test_evaluated":False,"test_signal_files_opened":0,"restoration":"Freshly measured rerun of the same fixed calibration; historical score was known before rerun"}
    base.write_json(OUT/"validation_results.json",result);print("CALIBRATED_VALIDATION",met,flush=True)

def development_features(indices):
    plan=base.get_plan();assert len(indices)==len(set(indices))
    assert set(indices).issubset(set(plan["train_indices"]+plan["validation_indices"]))
    arrays=[np.load(ROOT/f"{part}_features.npz") for part in ["train","validation"]]
    merged={k:np.concatenate([a[k] for a in arrays]) for k in ["X","y","groups","windows"]}
    positions=np.concatenate([np.flatnonzero(merged["groups"]==idx) for idx in indices])
    assert len(positions)==len(indices)*19
    return {k:v[positions] for k,v in merged.items()}

def fit_development_fold(fold,output_dir):
    verify_freeze();_,split,_,_=base.metadata();assert fold in range(1,6)
    outer=split["cv"][fold-1];assert outer["fold"]==fold and not set(outer["train"])&set(outer["test"])
    tr=development_features(outer["train"]);va=development_features(outer["test"])
    model,fit=fit_fixed_calibrated(tr["X"],tr["y"],tr["groups"]);p=predict_probabilities(model,va["X"])
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);joblib.dump(model,out/"model.joblib")
    write_predictions(out/"development_fold_probabilities.csv",va,p)
    base.write_json(out/"fit_provenance.json",{"role":"postselection development CV, not finaltest","outer_fold":fold,"training_indices":outer["train"],"evaluation_indices":outer["test"],"fit":fit})
    return model,p,fit

def fit_full_development(output_dir):
    verify_freeze();_,split,_,_=base.metadata();tr=development_features(split["holdout"]["train"])
    assert tr["X"].shape==(3781,39)
    model,fit=fit_fixed_calibrated(tr["X"],tr["y"],tr["groups"]);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/"model.joblib")
    base.write_json(out/"fit_provenance.json",{"role":"full199 development training only; test not evaluated","training_indices":split["holdout"]["train"],"fit":fit})
    return model,fit

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["freeze","validate"])
    {"freeze":freeze,"validate":run_validation}[parser.parse_args().stage]()
