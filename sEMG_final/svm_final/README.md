# 최종 선택: 39-feature calibrated RBF SVM

최종 제출 모델은 **39-feature + StandardScaler + RBF SVM + sigmoid calibration**이다. 저장된 과거 예측은 950개 창 중 887개를 맞혀 **93.37%**를 기록했다. 이 수치는 이미 여러 실험에서 재사용한 holdout의 탐색적 결과이며, 새로운 독립 평가나 현장 성능을 뜻하지 않는다.

이 폴더는 기존 고정 레시피의 휴대 가능한 재현 코드와 검증 근거를 모은 것이다. 이번 제출 정리에서는 **재학습·튜닝·원시 신호 평가를 실행하지 않았다.** 새 실행 래퍼를 과거 학습에 사용된 원본 실행 코드라고 주장하지 않는다. 원시 데이터, 특징 캐시, 학습 모델 바이너리는 포함하지 않는다. 상위 `results45/`의 기존 딥러닝 실험은 별도 비교 자료로 보존한다.

## 저장된 결과 검증

Python 3.12 환경을 권장한다. 아래 명령은 이 폴더에서 실행한다. 설치 버전은 과거 기록과 동일하게 고정되어 있다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python audit.py
python -m unittest -v test_audit.py
```

상위 `sEMG_final/`에서 `python svm_final/audit.py`로도 실행할 수 있다. 보고서 생성기는 `from svm_final.audit import audit; audit()`를 호출해 같은 검사를 수행할 수 있다. 감사는 저장된 CSV의 지표를 다시 계산하고, source SHA-256·분할·시행별 19개 창·확률 합·source provenance를 검사한다. 모델을 불러오거나 학습하지 않는다. 테스트는 calibration의 시행 경계 및 과거 fold와의 정확한 일치와, 합성 신호에서 원본 함수와의 수치 동일성도 확인한다. Python의 assert 검사를 비활성화하는 `-O`는 사용하지 않는다.

### 과거 재사용 holdout: window 단위

| Accuracy | Macro precision | Macro recall | Macro F1 |
|---:|---:|---:|---:|
| 93.368421% | 93.451498% | 93.368421% | 93.398076% |

- 50시행 × 19창 = 950창, 정답 887개
- 창은 300 samples(300 ms), hop 150 samples(150 ms), 시행 내 중첩
- 최대 비대각 오분류: 실제 E → 예측 B, 14창. `confusion.png`의 빨간 테두리
- 3초 시행의 19개 확률을 평균한 별도 지표는 50/50 정답이다. 이를 300 ms 단일창 성능이나 현장 100% 성공률로 해석하지 않는다
- 근거: `evidence/final_predictions.csv`, `evidence/final_metrics.json`

### 개발 5-fold: 선택 이후 안정성 확인

| Fold | 정답/창 | Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 1 | 713/760 | 93.815789% | 93.811525% |
| 2 | 683/760 | 89.868421% | 89.885481% |
| 3 | 707/760 | 93.026316% | 93.030366% |
| 4 | 707/760 | 93.026316% | 93.008711% |
| 5 | 677/741 | 91.363023% | 91.115180% |

동일 가중 fold 평균 Accuracy는 **92.219973%**, population SD(ddof=0)는 **1.422053 percentage points**이다. Sample SD(ddof=1)는 1.589903 pp이다. Macro F1 평균은 92.170252%, population SD는 1.447062 pp이다. Pooled OOF Accuracy는 3,487/3,781 = 92.224279%로, fold 평균과 구분한다.

개발 199시행 내부의 고정 5-fold이며 독립 test나 nested model-selection 성능 추정이 아니다. 분할 및 순서는 다시 뽑지 않고 `evidence/splits.json`을 그대로 사용한다. 근거는 `evidence/development_oof_predictions.csv`와 `evidence/completion.json`이다.

## 고정 전처리·모델

1. 데이터는 공개 저장소의 고정 commit `adb7955f4416165c88e4111af6f8fdafd416209c` 및 `evidence/manifest.csv`의 249시행을 사용한다. 클래스 A–E는 등록된 5명이다. 파일 해시와 float64로 해석한 신호 해시를 모두 검사한다.
2. 입력: 1,000 Hz, 3,000 samples × 2 channels(채널 3/4). 이미 필터링된 CSV에 60 Hz notch(Q=30), 4차 20–499 Hz Butterworth를 추가 적용하며 두 단계 모두 시행 전체에 `filtfilt`한다.
3. 300-sample 창, 150-sample hop으로 시행당 19창. 각 창에서 채널별 평균 제거. 창 min-max나 CWT는 사용하지 않는다.
4. 총 39특징: 채널별 log RMS·log MAV의 4 amplitude 특징 + 채널별 17 shape 특징과 채널 간 상관의 35 shape 특징.
5. Shape: RMS 정규화 후 Hann periodogram(nfft=300, detrend=False); 20–500 Hz의 상대 전력에서 20/50/100/150/250/350/500 Hz 경계의 6 log-band powers, mean/median frequency, spectral entropy, waveform length, MAV/RMS, skewness, excess kurtosis, zero crossing, slope-sign change, lag 1/2 값. 정확한 식·정의·배열 순서는 `features.py`가 기준이다.
6. `StandardScaler` → `SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, tol=1e-3, cache_size=256, decision_function_shape="ovr")`; class weighting 없음. Scaler는 각 학습 부분집합 안에서만 fit한다.
7. `CalibratedClassifierCV(method="sigmoid", ensemble=False, n_jobs=1)`; seed 2026의 3개 trial-grouped stratified fold로 학습 내 OOF score를 만들고 full-training base estimator를 다시 fit한다. 창을 독립 표본처럼 무작위 분할하지 않는다. 예측 클래스는 A–E 순서 확률의 argmax다.

과거에는 18개 레시피 비교 후 고정 calibration 변형 1개가 추가되었다. 이 폴더는 선택된 레시피만 실행하며 C/gamma/특징/calibration 재탐색 기능이 없다.

## 선택적 재실행: 원시 데이터를 별도로 준비한 경우만

아래는 재현을 원하는 사용자가 추후 실행할 명령이다. 이번 패키징에서 수행한 학습 기록이 아니다. 새 출력 디렉터리만 허용하며 보존된 evidence를 덮어쓰지 않는다. 정확한 파일을 구하지 못하거나 결과가 다르면 원인을 확인해야 하며, 과거 점수에 맞추기 위해 튜닝하지 않는다.

```bash
git clone https://github.com/sea3551/palm-sEMG-doorknob-filtered.git data
git -C data checkout adb7955f4416165c88e4111af6f8fdafd416209c
python reproduce.py cv --data-dir data/data --output rerun/cv
python reproduce.py fit --data-dir data/data --output rerun/fit
```

`cv`는 개발 199시행만 읽고 저장된 outer 5-fold마다 동일 레시피를 적용한다. `fit`도 개발 199시행만 읽고 full-development 모델과 provenance를 생성한다. 배열 순서·calibration fold는 과거 기록과 동일하다. CPU ID를 강제하지 않아 다른 컴퓨터에서도 사용할 수 있으나, numerical-library thread 수는 1로 설정한다. 실행 시간·모델 직렬화 해시는 환경 영향을 받을 수 있으며 과거와의 일치를 보장하지 않는다.

이미 재사용된 50시행을 다시 채점하려면 명시적 별도 명령이 필요하다. 이는 새로운 독립 평가가 아니다.

```bash
python reproduce.py evaluate --data-dir data/data --model rerun/fit/model.joblib --output rerun/reused_holdout --acknowledge-reused-holdout
```

`joblib`는 pickle 기반이다. 직접 생성한 신뢰할 수 있는 모델만 지정한다. 모델 해시·학습 인덱스·코드/분할 해시도 검사한다.

그림만 다시 만들려면 선택적으로 `python -m pip install matplotlib==3.10.8` 후 `python draw_confusion.py --output confusion_rerender.png`를 실행한다. 보존 그림은 감사 대상이므로 새 파일로 출력한다.

## Provenance와 한계

- `evidence/historical_source/`: 보관되어 있던 복구 실행 코드 3개를 byte-identical 복사. 여기서 필요한 feature 및 calibration 함수를 새 portable 코드로 추출했다. 이 보관 코드는 CPU-affinity, 옛 경로 및 중간 산출물에 의존하므로 직접 실행용이 아니다.
- `evidence/predeclared_plan.json`, `fixed_calibration_plan.json`, `frozen_classical_plan.json`: 과거 freeze 근거. 첫 파일에 원본 실행 환경 소실과 2026-10-04 복구 사실이 명시되어 있다. 보관 자료가 복구본임을 숨기지 않는다.
- `evidence/completion.json`: 2026-10-04의 고정 개발 CV/full-fit 기록. `evidence/final_metrics.json`과 최종 CSV는 2026-10-06의 별도 최종 평가에서 가져왔다. 전자는 원본 summary에서 SVM 및 공통 한계 항목만 추출했다.
- `evidence/provenance.json`: 각 보존 파일의 출처와 원본 SHA-256. 개발 OOF CSV는 5개 원본 CSV를 fold 열과 함께 연결했고, 감사가 각 원본 CSV를 재구성해 해시도 확인한다. `SHA256SUMS`는 이 패키지 파일의 무결성 기준이다. 로컬 `.gitattributes`는 CSV의 원래 줄바꿈을 유지해 Git checkout 시 해시가 달라지는 것을 막는다.
- 과거 full-development 모델은 39특징, 1,440 support vectors, 642,444 bytes였다. 기록된 full fit 0.751113275초는 **cached features 이후의 fitting만**, model-only inference 0.817736220 ms/window는 **precomputed features 이후의 prediction만**이다. 서로 다른 범위의 과거 측정이며 현재 실행 시간이나 end-to-end 지연을 나타내지 않는다.
- 5명·단일 세션·동일 등록자 분류다. 새로운 사람, 다른 날짜, 전극 위치 변화, open-set 거부 성능은 검증하지 않았다. Amplitude 특징은 피부/전극/세션 조건의 영향을 받을 수 있다.
- 시행 전체의 비인과적 `filtfilt` 때문에 정확히 같은 첫 창 특징에도 3초 기록이 필요하다. 실시간 300 ms 출입 결정이나 causal streaming 성능으로 주장할 수 없다.
- 재사용 holdout와 선택 이후 CV는 성능 선택 편향을 제거하지 못한다. 실제 적용 전 별도 수집한 독립 세션과 운영 조건의 평가가 필요하다.
