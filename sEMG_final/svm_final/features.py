"""Fixed feature formulas extracted from retained historical source.

See evidence/provenance.json. This portable packaging was not the historical
executed training entrypoint. No source algorithms or feature order are tuned.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, periodogram

LABELS = list("ABCDE")

WIN, HOP = 300, 150

BANDS = [(20,50),(50,100),(100,150),(150,250),(250,350),(350,500)]

AMP_NAMES = [f"ch{ch}_{stat}" for ch in [3,4] for stat in ["log_rms","log_mav"]]

CHANNEL_SHAPE_NAMES = [f"log_relative_power_{lo}_{hi}" for lo,hi in BANDS]+[
    "mean_frequency_hz","median_frequency_hz","spectral_entropy",
    "waveform_length_per_sample_over_rms","mav_over_rms","skewness",
    "excess_kurtosis","zero_crossing_fraction","slope_sign_change_fraction",
    "lag1_correlation","lag2_correlation"]

SHAPE_NAMES = [f"ch{ch}_{stat}" for ch in [3,4] for stat in CHANNEL_SHAPE_NAMES]+["cross_channel_correlation"]

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
