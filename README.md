# 손바닥 sEMG 사용자 식별

## 최종 과제

최종 선택은 **39특징 보정 SVM 단독**이다. 기존 재사용 시험집합 50시행의 950창 중 887개를 맞혀 **Accuracy 93.37%, macro F1 93.40%**를 기록했다. 새로운 독립시험이나 이번에 새로 수행한 SVM 튜닝 결과는 아니다.

- [실행 방법·모델 비교·최종 결과](sEMG_final/README.md)
- [최종 보고서 PDF](sEMG_final/results45/summary/report.pdf)
- [SVM 재현 코드·예측 검산](sEMG_final/svm_final/README.md)
- [최종 선택 기록](sEMG_final/final_selection.json)
- [필수 딥러닝 결과 비교표](sEMG_final/results45/summary/comparison.csv)

수업 필수 항목인 DenseNet161·ResNet18·SimpleCNN의 각 3seed 45epoch 학습, DenseNet 5-fold, BN 변형, 4가지 지표·혼동행렬·오류 분석·연산비용·한계·재현 정보를 함께 제시했다. 해당 딥러닝 비교군의 최고 평균 Accuracy는 DenseNet161+BN 88.95%이며, 최종 SVM의 단일 모델 결과와 구분한다. 논문 94%를 정확히 재현했다고 주장하지 않는다. 생성형 AI를 코드 작성·실험·검증·보고서 정리에 사용했다.

실행은 sEMG_final 폴더에서 시작한다. 아래의 1주차 제출 자료는 보존했다.

# 손바닥 sEMG 사용자 식별 — 1주차 실습

문손잡이 회전 중 기록한 손바닥 sEMG 공개 데이터의 구조와 파형을 확인한 수업 실습입니다.

## 제출 파일
- [PDF 보고서](submission/week1_report.pdf): 환경 확인, 데이터 요약, 피험자 5명의 파형, 관찰 3개, 선행연구 표와 차이점
- [실행 결과 포함 노트북](week1_report.ipynb)
- [노트북 HTML](submission/week1_report.html): Jupyter 없이 결과 읽기
- GitHub 제출 링크: https://github.com/KyunghoCha/semg-auth-coursework

저장소는 제출 전 검토를 위해 비공개로 생성했습니다. 링크를 제출할 때는 평가자가 접근할 수 있어야 합니다.

## 결과와 범위
- CSV 250개: A–E 각 50개, 모든 파일 (3000, 2), NaN·무한대 없음
- 파일명상 1–50번 시행을 피험자별로 확인
- 1,000 Hz, 3초, 두 채널; 파지 1초 → 회전 1초 → 정지 1초
- 대표 파형은 각 피험자의 1번 시행이며 폴더와 시행 번호로 선택
- E의 38번과 50번은 내용이 완전히 같아 고유 파일 내용은 249개
- 원자료는 수정하지 않음; 공개 데이터의 기존 필터링 외 추가 필터링·정규화 없음
- 관찰 메모는 대표 5개 시행에 한정; 이번 주 실습에서는 모델을 학습하지 않음
- 표의 94.00%는 대상 논문 Table 3의 시험 정확도

## 환경
WSL Ubuntu 22.04 / Anaconda semg / Python 3.12.14입니다.
PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, RTX 3070 Laptop GPU에서
GPU 인식과 간단한 CUDA 텐서 연산을 확인했습니다.

정확한 실행 기록:
- [환경 출력](submission/environment_check.txt)
- [환경 버전](submission/environment.json)
- [파일별 SHA-256 및 검사 결과](submission/data_manifest.csv)

환경 확인 이미지는 실제 표준출력을 이미지로 저장한 것입니다.

## 재현
아래 명령은 WSL에서 실행합니다. 이미 semg 환경을 사용 중이면 환경 생성과 설치를 반복하지 않아도 됩니다.

```bash
git clone https://github.com/KyunghoCha/semg-auth-coursework.git
cd semg-auth-coursework

conda create -n semg python=3.12 pip -y
conda activate semg
python -m pip install -r requirements.txt
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/sea3551/palm-sEMG-doorknob-filtered.git data
git -C data checkout adb7955f4416165c88e4111af6f8fdafd416209c

python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=180 week1_report.ipynb
python scripts/build_week1_pdf.py
python -m jupyter nbconvert --to html --output-dir submission week1_report.ipynb
```

그림의 한글 표시에는 NanumGothic 등 한글 글꼴이 필요합니다.
Ubuntu에 해당 글꼴이 없다면 다음 명령으로 설치할 수 있습니다.

```bash
sudo apt-get install fonts-nanum
```

GPU가 없는 환경에서는 환경 점검 셀의 CUDA 검증이 실패하도록 구성했습니다.
저장된 노트북 출력과 PDF는 재실행 없이 읽을 수 있습니다.
VS Code의 WSL 연결 창에서 폴더를 열고 노트북 커널을 semg로 선택합니다.

노트북 셀을 수정해 실습할 수 있습니다. 초기 구성 자체를 재생성할 때만 아래 명령을 실행합니다.
이 명령은 노트북을 덮어씁니다.

```bash
python scripts/build_week1_notebook.py
```

## 파일 구성
- **week2/**: 기존 실습 스크립트 보존. 폴더 이름과 관계없이 이 보고서는 1주차 제출 범위입니다.
- **scripts/**: 노트북 생성 및 PDF 작성 코드
- **submission/**: 보고서와 실행 로그, 요약표, 파형 이미지
- **data/**: 원저자 공개 데이터 저장소. Git 추적에서 제외했으며 위 명령으로 별도 다운로드합니다.
- 루트의 기존 signal_example 이미지들은 로컬에 보존했습니다.

## 출처
1. Shin, Y., Kim, J., & Choi, S.-I. (2026).
   *Palm sEMG-based user identification during doorknob rotation using a convolutional neural network*.
   Scientific Reports, 16, 22244.
   https://doi.org/10.1038/s41598-026-46294-3
2. Data © 2025 Yeonjung Shin — CC BY 4.0.
   https://github.com/sea3551/palm-sEMG-doorknob-filtered
   https://creativecommons.org/licenses/by/4.0/
3. 수업 자료: Step1_1주차_강의와실습.pdf, 14쪽 및 17쪽.
   원논문과 수업 자료 PDF는 이 저장소에 재배포하지 않았습니다.

선행연구 표는 대상 논문의 Table 1을 재구성한 2차 인용이며,
각 선행연구를 별도로 재현하거나 동일 조건에서 비교한 결과가 아닙니다.
파형은 위 공개 데이터의 대표 시행을 시각화한 것입니다.

