from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 이 스크립트가 들어 있는 week2 폴더의 상위 = 프로젝트 폴더
project_dir = Path(__file__).resolve().parent.parent
files = sorted((project_dir / "data").rglob("*.csv"))

num=200

x = np.loadtxt(files[num], delimiter=',', skiprows=1)
if x.shape[0] < x.shape[1]: x = x.T # (시간, 채널) 로 통일
t = np.arange(len(x)) / 1000.0 # 1000 Hz -> 초 단위

fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
for ch in range(2):
    ax[ch].plot(t, x[:, ch], lw=0.6)
    ax[ch].set_ylabel(f'ch{ch+1}')
for a in ax:
    a.axvline(1.0, color='r', ls='--')
    a.axvline(2.0, color='r', ls='--')
ax[1].set_xlabel('time (s)')
plt.tight_layout(); plt.savefig(f'signal_example{num}.png', dpi=120)