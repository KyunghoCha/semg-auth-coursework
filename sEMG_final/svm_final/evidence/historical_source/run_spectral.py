"""Restored frozen sEMG diagnostic: training and development validation only.

Reconstructed from the original task's retained source after executor loss.
The scientific protocol is unchanged; this restoration has a new source hash.
No raw holdout loader or holdout evaluation is implemented.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time
import warnings
from datetime import datetime, timezone
import joblib
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, periodogram
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SEMG_DATA_DIR", ROOT.parent/"semg_improvement/data/data"))
MANIFEST = ROOT/"source_control/manifest.csv"
SPLITS = ROOT/"source_control/splits.json"
DATA_COMMIT = "adb7955f4416165c88e4111af6f8fdafd416209c"
LABELS = list("ABCDE")
WIN, HOP = 300, 150
BANDS = [(20,50),(50,100),(100,150),(150,250),(250,350),(350,500)]
FAMILIES = ["amplitude","shape","combined"]
EXPECTED_VALIDATION = [74,37,68,117,31,245,237,235,193,241,113,210,91,53,214,55,203,123,119,30,182,100,3,207,199,131,40,194,160,90,27,14,25,172,108,195,139,70,187,64]
VERSIONS = {"numpy":"2.3.5","scipy":"1.17.0","scikit-learn":"1.8.0","joblib":"1.5.3"}
AMP_NAMES = [f"ch{ch}_{stat}" for ch in [3,4] for stat in ["log_rms","log_mav"]]
CHANNEL_SHAPE_NAMES = [f"log_relative_power_{lo}_{hi}" for lo,hi in BANDS]+[
    "mean_frequency_hz","median_frequency_hz","spectral_entropy",
    "waveform_length_per_sample_over_rms","mav_over_rms","skewness",
    "excess_kurtosis","zero_crossing_fraction","slope_sign_change_fraction",
    "lag1_correlation","lag2_correlation"]
SHAPE_NAMES = [f"ch{ch}_{stat}" for ch in [3,4] for stat in CHANNEL_SHAPE_NAMES]+["cross_channel_correlation"]

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    tmp.replace(path)

def write_csv(path,rows):
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def metadata():
    rows=list(csv.DictReader(MANIFEST.open()))
    split=json.loads(SPLITS.read_text())
    assert len(rows)==249
    assert digest(MANIFEST)=="749a6a15a22630447b26a7800b498eb54690a6e1cfbd3bbf961b7a0b9b51aad6"
    assert digest(SPLITS)=="354fb9b91aeeb039d596d2f636832163621dc1c824f770f35ba479aea83b2f30"
    for i,row in enumerate(rows):
        assert int(row["index"])==i
    development=split["holdout"]["train"]
    train,validation=train_test_split(development,test_size=.2,random_state=123,stratify=[rows[i]["subject"] for i in development])
    assert validation==EXPECTED_VALIDATION and len(train)==159
    assert not set(train)&set(validation)
    assert set(train+validation)==set(development)
    assert not set(development)&set(split["holdout"]["test"])
    assert len({rows[i]["signal_sha256"] for i in development})==199
    return rows,split,train,validation

def check_cpu_versions():
    assert set(os.sched_getaffinity(0))=={8},"Use taskset -c 8"
    for package,version in VERSIONS.items():
        assert importlib.metadata.version(package)==version,(package,importlib.metadata.version(package))

def freeze():
    check_cpu_versions()
    assert not (ROOT/"predeclared_plan.json").exists(),"Already frozen"
    rows,split,train,validation=metadata()
    recipes=[{"family":family,"model":model,"C":c} for family in FAMILIES for model in ["logistic","rbf_svm"] for c in [.1,1.,10.]]
    folds=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2026).split(train,[rows[i]["subject"] for i in train]),1):
        folds.append({"fold":fold,"train_trial_indices":[train[j] for j in tr],"validation_trial_indices":[train[j] for j in va]})
    plan={"frozen_utc":utc(),"status":"restoration of the previously frozen scientific protocol; no new search or tuning",
          "original_plan_sha256":"015c7eb27fa6d4495ea8b1074127a6d48dfb86f18b35c974cc8f384385431c8c",
          "restoration_note":"Original local files lost when executor was replaced. Formulas, recipe order, splits, calibration and folds recovered from retained task source. Current code hashes intentionally differ; historical scores are reference targets, not rerun measurements.",
          "dataset_commit":DATA_COMMIT,"versions":VERSIONS,"train_indices":train,"validation_indices":validation,
          "train_paths":[rows[i]["path"] for i in train],"validation_paths":[rows[i]["path"] for i in validation],
          "source_sha256":{p.name:digest(p) for p in [MANIFEST,SPLITS]},
          "code_sha256":{p.name:digest(p) for p in sorted(ROOT.glob("*.py"))},
          "recipes":recipes,"inner_folds":folds,"features":{"amplitude":AMP_NAMES,"shape":SHAPE_NAMES,"combined":AMP_NAMES+SHAPE_NAMES},
          "preprocessing":"Whole trial60Hz notchQ30 then fourth-order20-499Hz Butterworth,filtfilt,fs1000;19windows300samples/hop150; per-channel window mean removed; no minmax",
          "spectrum":"Hann periodogram nfft300,detrend=False; relative20-500Hz power; bands20/50/100/150/250/350/500",
          "scaler":"StandardScaler inside fold-training pipeline; final scaler fitted only on assigned training subset",
          "selection":"Within family pooledOOF windowaccuracy, then macroF1, then earlier recipe order. Three winners frozen before validation.",
          "model_settings":"Logistic lbfgs,L2,max_iter2000,tol1e-6,seed2026; RBF-SVC gamma=scale,probability=False,tol1e-3,cache256MB,OVR; no class weighting",
          "calibration_fixed":"Previously approved single postdiagnostic variant: combined39 RBF C1; CalibratedClassifierCV sigmoid,ensemble=False,n_jobs1, explicit3trial-grouped folds seed2026",
          "classical_finalist_fixed":"Same calibrated variant on exact5developmentfolds then full199development fit; no variant selection based on these folds",
          "uncertainty":"10000 stratified whole-trial bootstrap samples seed2026; descriptive conditional intervals; no selection-adjusted inference",
          "trial_metrics":"Majority of19window predictions; tie smallest classindex. Probability models additionally report trial mean-probability. Separate from window metrics.",
          "historical_reference_only":{"raw_combined_correct":704,"calibrated_correct":705,"validation_windows":760,"cv_fold_correct":[713,683,707,707,677],"cv_fold_n":[760,760,760,760,741],"cv_mean":0.9221997300944669,"cv_population_sd":0.014220525706937569},
          "test_signal_reads_allowed":False,"test_evaluation_allowed":False,"resource":"CPU8 only,oneBLAS/OpenMPthread,noGPU,no paidcompute"}
    write_json(ROOT/"predeclared_plan.json",plan)
    print("RESTORED_PROTOCOL_FROZEN",utc(),digest(ROOT/"predeclared_plan.json"),flush=True)

def get_plan():
    check_cpu_versions()
    plan=json.loads((ROOT/"predeclared_plan.json").read_text())
    for name,sha in plan["code_sha256"].items():
        assert digest(ROOT/name)==sha,("Frozen code changed",name)
    for name,sha in plan["source_sha256"].items():
        assert digest(ROOT/"source_control"/name)==sha
    return plan

def feature_window(window):
    centered = window - window.mean(axis=0, keepdims=True)
    amp, shape = [], []
    for ch in range(2):
        x = centered[:, ch]
        rms = max(float(np.sqrt(np.mean(x*x))), 1e-24)
        mav = max(float(np.mean(np.abs(x))), 1e-24)
        z = x / rms
        amp.extend([np.log(rms), np.log(mav)])
        freq, psd = periodogram(z, fs=1000, window="hann", nfft=300, detrend=False)
        valid = (freq >= 20) & (freq <= 500)
        f, power = freq[valid], psd[valid]
        prob = power / max(float(power.sum()), 1e-24)
        bp = [prob[(f >= lo) & ((f < hi) if hi < 500 else (f <= hi))].sum() for lo, hi in BANDS]
        dz = np.diff(z)
        shape.extend(np.log(np.asarray(bp) + 1e-8).tolist())
        shape.extend([float((f*prob).sum()), float(f[np.searchsorted(np.cumsum(prob), 0.5)]),
                      float(-(prob*np.log(np.maximum(prob, 1e-24))).sum()/np.log(len(prob))),
                      float(np.abs(dz).mean()), mav/rms, float(np.mean(z**3)), float(np.mean(z**4)-3),
                      float(np.mean(z[:-1]*z[1:] < 0)), float(np.mean(dz[:-1]*dz[1:] < 0)),
                      float(np.mean(z[:-1]*z[1:])), float(np.mean(z[:-2]*z[2:]))])
    sd = np.sqrt(np.mean(centered*centered, axis=0))
    shape.append(float(np.mean(centered[:, 0]*centered[:, 1])/max(float(np.prod(sd)), 1e-24)))
    result = np.array(amp + shape, dtype=np.float64)
    assert result.shape == (39,) and np.isfinite(result).all()
    return result

def transform_trial(signal):
    signal=np.asarray(signal,dtype=np.float64)
    assert signal.shape==(3000,2) and np.isfinite(signal).all()
    bn,an=iirnotch(60,30,fs=1000)
    b,a=butter(4,[20,499],btype="bandpass",fs=1000)
    filtered=filtfilt(b,a,filtfilt(bn,an,signal,axis=0),axis=0)
    return np.stack([feature_window(filtered[start:start+WIN]) for start in range(0,3000-WIN+1,HOP)])

def extract(partition):
    plan=get_plan();rows,split,train,validation=metadata()
    assert partition in ["train","validation"]
    allowed=plan[partition+"_indices"]
    X,y,groups,windows,audit=[],[],[],[],[]
    for idx in allowed:
        row=rows[idx];path=DATA/row["path"]
        assert path.resolve().is_relative_to(DATA.resolve())
        assert digest(path)==row["sha256"]
        signal=np.loadtxt(path,delimiter=",",skiprows=1,dtype=np.float64)
        assert hashlib.sha256(signal.astype("<f8").tobytes()).hexdigest()==row["signal_sha256"]
        features=transform_trial(signal)
        X.extend(features);y.extend([LABELS.index(row["subject"])]*19)
        groups.extend([idx]*19);windows.extend(range(19));audit.append(row)
    ds={"X":np.asarray(X),"y":np.asarray(y),"groups":np.asarray(groups),"windows":np.asarray(windows)}
    assert ds["X"].shape==(len(allowed)*19,39)
    write_json(ROOT/f"{partition}_files_opened.json",audit)
    np.savez_compressed(ROOT/f"{partition}_features.npz",**ds)
    return ds

def choose_features(x,family):
    return x[:,:4] if family=="amplitude" else x[:,4:] if family=="shape" else x

def estimator(recipe):
    if recipe["model"]=="logistic":
        model=LogisticRegression(C=recipe["C"],solver="lbfgs",max_iter=2000,tol=1e-6,random_state=2026)
    else:
        assert recipe["model"]=="rbf_svm"
        model=SVC(C=recipe["C"],kernel="rbf",gamma="scale",probability=False,cache_size=256,tol=1e-3,decision_function_shape="ovr")
    return make_pipeline(StandardScaler(),model)

def measures(y,pred):
    return {"accuracy":float(accuracy_score(y,pred)),"f1_macro":float(f1_score(y,pred,labels=range(5),average="macro",zero_division=0)),
            "correct":int(np.sum(y==pred)),"n":len(y),"confusion_matrix":confusion_matrix(y,pred,labels=range(5)).tolist()}

def trial_vote(y,pred):
    yy,pp=y.reshape(-1,19),pred.reshape(-1,19)
    assert np.all(yy==yy[:,:1])
    return yy[:,0],np.array([np.bincount(row,minlength=5).argmax() for row in pp])

def train():
    plan=get_plan()
    assert not (ROOT/"selected_recipes.json").exists() and not (ROOT/"validation_features.npz").exists()
    started=time.perf_counter();ds=extract("train");summaries=[];fold_rows=[];selected={}
    for number,recipe in enumerate(plan["recipes"]):
        x=choose_features(ds["X"],recipe["family"]);out=np.full(len(ds["y"]),-1);seen=np.zeros(len(out),int)
        for fold in plan["inner_folds"]:
            tr=np.isin(ds["groups"],fold["train_trial_indices"]);va=np.isin(ds["groups"],fold["validation_trial_indices"])
            assert not np.any(tr&va) and np.all(tr|va)
            model=estimator(recipe);begin=time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error",ConvergenceWarning);model.fit(x[tr],ds["y"][tr])
            assert int(model[0].n_samples_seen_)==int(tr.sum())
            pred=model.predict(x[va]);out[va]=pred;seen[va]+=1;met=measures(ds["y"][va],pred)
            fold_rows.append({"recipe_no":number,**recipe,"fold":fold["fold"],"accuracy":met["accuracy"],"f1_macro":met["f1_macro"],"seconds":time.perf_counter()-begin})
            write_csv(ROOT/"inner_cv_folds.csv",fold_rows)
        assert np.all(seen==1)
        row={"recipe_no":number,**recipe,**measures(ds["y"],out)};summaries.append(row)
        write_json(ROOT/"inner_cv_summary.json",summaries)
        np.savez_compressed(ROOT/f"inner_oof_recipe{number:02d}.npz",pred=out,y=ds["y"],groups=ds["groups"])
        print("INNER",number,recipe,row["accuracy"],flush=True)
    for family in FAMILIES:
        winner=max([r for r in summaries if r["family"]==family],key=lambda r:(r["accuracy"],r["f1_macro"],-r["recipe_no"]))
        recipe={k:winner[k] for k in ["family","model","C"]};x=choose_features(ds["X"],family);model=estimator(recipe)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning);model.fit(x,ds["y"])
        pred=model.predict(x);path=ROOT/f"model_{family}.joblib";joblib.dump(model,path)
        selected[family]={"recipe":recipe,"inner_oof":winner,"training_resubstitution_window":measures(ds["y"],pred),
                          "training_resubstitution_trial_vote":measures(*trial_vote(ds["y"],pred)),"model_sha256":digest(path)}
    result={"selected_utc":utc(),"plan_sha256":digest(ROOT/"predeclared_plan.json"),"selected":selected,"training_seconds":time.perf_counter()-started,"validation_accessed_before_selection":False}
    write_json(ROOT/"selected_recipes.json",result);print("WINNERS_FROZEN",utc(),flush=True)

def write_predictions(path,ds,pred,prob=None):
    manifest,_,_,_=metadata();rows=[]
    for j,(y,p,g,w) in enumerate(zip(ds["y"],pred,ds["groups"],ds["windows"])):
        row={"path":manifest[int(g)]["path"],"trial_index":int(g),"window":int(w),"true":LABELS[int(y)],"predicted":LABELS[int(p)]}
        if prob is not None:
            row.update({"prob_"+label:float(prob[j,k]) for k,label in enumerate(LABELS)})
        rows.append(row)
    write_csv(path,rows)

def bootstrap_samples(y):
    ytrial=y.reshape(-1,19)[:,0];rng=np.random.default_rng(2026)
    return np.concatenate([rng.choice(np.flatnonzero(ytrial==c),size=(10000,np.sum(ytrial==c)),replace=True) for c in range(5)],axis=1)

def validate():
    get_plan();assert not (ROOT/"validation_results.json").exists()
    winners=json.loads((ROOT/"selected_recipes.json").read_text());ds=extract("validation");result={};preds={}
    resamples=bootstrap_samples(ds["y"]);boot={}
    for family,sel in winners["selected"].items():
        path=ROOT/f"model_{family}.joblib";assert digest(path)==sel["model_sha256"]
        pred=joblib.load(path).predict(choose_features(ds["X"],family));preds[family]=pred
        result[family]={"recipe":sel["recipe"],"window":measures(ds["y"],pred),"trial_majority_vote":measures(*trial_vote(ds["y"],pred)),
                        "training_resubstitution_window":sel["training_resubstitution_window"],"inner_oof_window":sel["inner_oof"]}
        write_predictions(ROOT/f"validation_predictions_{family}.csv",ds,pred)
        boot[family]=(pred==ds["y"]).reshape(-1,19).mean(1)[resamples].mean(1)
        result[family]["descriptive_window_accuracy_ci95"]=np.quantile(boot[family],[.025,.975]).tolist()
        yt,pt=trial_vote(ds["y"],pred)
        result[family]["descriptive_trial_vote_accuracy_ci95"]=np.quantile((yt==pt)[resamples].mean(1),[.025,.975]).tolist()
        print("VALIDATION",family,result[family]["window"],flush=True)
    pairs=[{"comparison":f"{b} minus {a}","window_accuracy_difference":result[b]["window"]["accuracy"]-result[a]["window"]["accuracy"],
            "descriptive_ci95":np.quantile(boot[b]-boot[a],[.025,.975]).tolist()} for a,b in itertools.combinations(FAMILIES,2)]
    write_json(ROOT/"validation_results.json",{"completed_utc":utc(),"results":result,"paired_differences":pairs,"test_evaluated":False,"test_signal_files_opened":0})

def selftest():
    check_cpu_versions();metadata();x=np.random.default_rng(2026).normal(size=(300,2))
    first=feature_window(x);gained=feature_window(x*np.array([4.,.3])+np.array([100.,-40.]))
    assert np.allclose(first[4:],gained[4:],atol=1e-10)
    assert np.allclose(gained[:4]-first[:4],np.log([4.,4.,.3,.3]))
    print("SELFTEST_PASS: feature invariance and exact frozen split metadata; no source signals loaded",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["selftest","freeze","train","validate"])
    stage=parser.parse_args().stage
    {"selftest":selftest,"freeze":freeze,"train":train,"validate":validate}[stage]()
