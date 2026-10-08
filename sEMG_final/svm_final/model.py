"""Portable wrapper for the single frozen, calibrated combined-39 RBF SVM."""
import time
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

def build_model(y, groups):
    folds, provenance = grouped_calibration_folds(y, groups)
    pipeline = make_pipeline(StandardScaler(), SVC(
        C=1.0, kernel="rbf", gamma="scale", probability=False,
        cache_size=256, tol=1e-3, decision_function_shape="ovr"))
    return CalibratedClassifierCV(
        estimator=pipeline, method="sigmoid", cv=folds,
        n_jobs=1, ensemble=False), provenance


def fit_fixed(x, y, groups):
    if x.ndim != 2 or x.shape[1] != 39 or not np.isfinite(x).all():
        raise ValueError("Expected finite (n_windows, 39) feature matrix")
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("Feature, label, and group counts differ")
    model, folds = build_model(y, groups)
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start
    pipeline = model.calibrated_classifiers_[0].estimator
    assert len(model.calibrated_classifiers_) == 1
    assert np.array_equal(model.classes_, np.arange(5))
    assert int(pipeline[0].n_samples_seen_) == len(x)
    np.testing.assert_allclose(pipeline[0].mean_, x.mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(pipeline[0].var_, x.var(0), rtol=0, atol=1e-12)
    return model, {"calibration_folds": folds, "training_windows": len(x),
                   "training_trials": len(set(groups)), "fit_seconds": elapsed,
                   "train_only_scaler_verified": True}


def predict_probabilities(model, x):
    assert np.array_equal(model.classes_, np.arange(5))
    p = model.predict_proba(x)
    assert p.shape == (len(x), 5) and np.isfinite(p).all() and np.all(p >= 0)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-12, rtol=0)
    return p
