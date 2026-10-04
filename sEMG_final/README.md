# 손바닥 sEMG 식별

CWT로 등록자 A-E를 분류하는 Step 1 비교실험이다. 방법·혼동행렬·오류·한계는 [보고서](results45/summary/report.pdf)에 있다.

## 데이터와 방법

- 공개250시행 중 동일 신호 E50을 제외한249시행 사용
- 시행을 먼저 학습 199 / 시험 50으로 분할(seed 42), 이후300 ms 창·150 ms hop으로3781/950윈도우 생성
- 이미 필터링된 CSV에 수업의60 Hz notch(Q=30), 4차 20-499 Hz 필터를 추가 적용
- 창 전체 min-max, morl CWT 스케일 1-32, 두 채널 절댓값과 평균: 3×32×300
- 세 모델을 사전학습 없이 Adam 0.001·batch 16·45epoch로 seed 42/43/44 각각 학습
- DenseNet+BN은 같은 가중치에 학습 데이터만 이용한 BN 재보정 1회 추가. 개발검증에서 순증2/760창의 작은 차이로 선택
- 모델 구조: SimpleCNN은 두 합성곱(3→32→64)과 평균 pooling, ResNet18은 잔차 연결과 512→5 분류기, DenseNet161은 조밀 연결과2208→5 분류기
- DenseNet은 torchvision 기본 첫 합성곱과 dropout 없음. 공통 학습 설정은 수업 예제를 따랐으며 논문 구현의 미공개 세부까지 일치한다고 주장하지 않는다
- 최종 시험 50시행은 이전5epoch 결과를 본 뒤 재사용했다. DenseNet5-fold는199 개발시행 내부에서만 수행

분할은 [data_utils.py](data_utils.py)의 split_trials와 experiment.py의 make_plan에서 시행 단위로 먼저 수행한다. 창 생성은 그 이후다.

| 윈도우 | A | B | C | D | E | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| 학습 | 760 | 760 | 760 | 760 | 741 | 3781 |
| 시험 | 190 | 190 | 190 | 190 | 190 | 950 |

## 실제 결과

950윈도우(원본 50시행)의3회 평균(%). Precision·Recall·F1은 macro 평균이다.

| 모델 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| SimpleCNN | 62.63 | 62.70 | 62.63 | 62.08 |
| ResNet18 | 88.04 | 88.20 | 88.04 | 87.99 |
| DenseNet161 | 88.53 | 88.92 | 88.53 | 88.55 |
| DenseNet161+BN | 88.95 | 89.31 | 88.95 | 88.90 |

최고: DenseNet161+BN, 최저: SimpleCNN(평균 Accuracy 기준). 모델별 혼동행렬과 오류, fold별 값·평균±SD, 시간·파라미터는 보고서에 있다.

## 실행

Linux, Python 3.12, 가용 논리 CPU 최소 4개, flock/taskset이 필요하다. 아래 기본 명령은 논리 CPU 4개 1작업이다. 실제 기록은 가용 논리 CPU 9개에서 각 4개를 쓰는 두 작업을 병렬 실행했다. 전체 재학습에는 수 시간이 걸린다.

PDF 재생성에는 NanumGothic 글꼴이 필요하다. Ubuntu: sudo apt-get install fonts-nanum. 다른 환경은 assets/README.md를 참고한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
git clone https://github.com/sea3551/palm-sEMG-doorknob-filtered.git data
git -C data checkout adb7955f4416165c88e4111af6f8fdafd416209c
python -m unittest -v test_project.py test_summary45.py
mkdir -p rerun45
cp results45/frozen_plan.json rerun45/
python run_cpu45.py --output-root rerun45 --cache-dir cache45 --workers 1
python complete_evaluation45.py --run-root rerun45 --cache-dir cache45 --evaluate-test
python make_report45.py --run-root rerun45
```

완료 체크포인트는 재사용한다. 새 실험은 별도 출력 폴더와 사전 계획을 사용하며 시험 결과로 설정을 바꾸지 않는다. 재학습 없이 수치를 재검산하려면 python summarize45.py --run-root results45를 실행한다.

BN 선택검증 재현(선택):

```bash
python colab_run.py --job validate --model DenseNet161 --seed 42 --epochs 45 --device cpu --threads 4 --data-dir data/data --cache-dir cache45 --output-dir rerun_validation
python bn_recalibrate.py --run-dir rerun_validation --output-dir rerun_bn --data-dir data/data --cache-dir cache45 --threads 4
```

## 파일

| 파일 | 역할 |
|---|---|
| data_utils.py | CSV 검증·중복 제거·시행 단위 분할 |
| experiment.py | 필터·CWT·세 모델·결정론 설정의 공통 함수, 5epoch 기준실험 |
| colab_run.py | 단일45epoch 작업, 모델·Adam·RNG 체크포인트 복구 |
| run_cpu45.py | 고정된3모델×3seed와 DenseNet5-fold 학습; 시험 평가는 차단 |
| finalize45.py / complete_evaluation45.py | 고정 계획 검사, 순차 평가, DenseNet BN 변형 |
| summarize.py | 혼동행렬 그림 공통 함수 |
| summarize45.py / make_report45.py | 예측 재검산, 표·혼동행렬·PDF·README |
| bn_recalibrate.py | 개발검증에서 수행한 BN 후보 진단과 반복 일치 검사 |
| test_project.py / test_summary45.py | 중복·누수·분할·CWT·모델 출력·예측 파일 검사 |
| results45/ | 고정 계획·실제 loss·설정·예측·지표·보고서 |
| results45/development/ | BN 선택 당시의 비교 지표와 원본·재보정 예측 |
| assets/ | PDF 글꼴 설치 안내와 NanumGothic OFL 라이선스 |

데이터·CWT 캐시·대형 가중치는 Git에서 제외한다. 재학습은 공개 데이터를 내려받아 재생성한다.

## 한계와 출처

5명·단일 세션의 closed-set 결과이며 겹치는 창은 독립 시행이 아니다. 재사용 시험집합과 개발집합에서 선택한 BN 변형의 결과를 새로운 독립시험이나 논문94%의 정확한 재현으로 주장하지 않는다.

- 데이터: https://github.com/sea3551/palm-sEMG-doorknob-filtered · Data © 2025 Yeonjung Shin, CC BY 4.0
- 논문: https://doi.org/10.1038/s41598-026-46294-3
- BN 통계 갱신: https://docs.pytorch.org/docs/2.8/optim.html#torch.optim.swa_utils.update_bn
- 코드 구성: Step 1 2·3주차 강의 예제 기반
- 생성형 AI 사용: 코드 작성·실험·검증·보고서 정리. 결과는 실제 실행과 예측에서 계산
