"""Build the concise final PDF/README only after all22 evaluations verify."""
import argparse
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from data_utils import load_trials
from experiment import make_plan, preprocess, make_windows, normalize_windows, to_cwt, write_csv
from summarize45 import summarize

ROOT=Path(__file__).resolve().parent


def pct(value): return f'{100*value:.2f}'


def report_font(bold=False):
    """Use optional local fonts or the Ubuntu fonts-nanum package."""
    names = ['NanumGothic-Bold.ttf', 'NanumGothicBold.ttf'] if bold else ['NanumGothic-Regular.ttf', 'NanumGothic.ttf']
    directories = [ROOT/'assets', Path('/usr/share/fonts/truetype/nanum')]
    if os.environ.get('SEMG_FONT_DIR'):
        directories.insert(0, Path(os.environ['SEMG_FONT_DIR']))
    for directory in directories:
        for name in names:
            path = directory/name
            if path.is_file(): return path
    raise FileNotFoundError('Install fonts-nanum (sudo apt-get install fonts-nanum), or set SEMG_FONT_DIR to your NanumGothic font directory.')


def method_figures(out,data_dir):
    trials,_=load_trials(data_dir);plan=make_plan(trials)
    trial=trials[plan['holdout']['train'][0]];filtered=preprocess(trial.signal)
    f0,p0=welch(trial.signal[:,0],fs=1000,nperseg=512)
    f1,p1=welch(filtered[:,0],fs=1000,nperseg=512)
    fig,ax=plt.subplots(figsize=(6,2.8))
    ax.semilogy(f0,p0,label='Provided filtered CSV');ax.semilogy(f1,p1,label='After additional lecture filters')
    ax.axvline(60,color='red',linestyle='--',linewidth=.8)
    ax.set(xlim=(0,300),xlabel='Frequency (Hz)',ylabel='Power',title=trial.path)
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'filter_check.png',dpi=150);plt.close(fig)
    window=normalize_windows(make_windows(filtered))[0]
    fig,ax=plt.subplots(figsize=(6,2.8))
    im=ax.imshow(to_cwt(window)[0],aspect='auto',origin='lower',extent=[0,.3,1,32],cmap='viridis')
    ax.set(xlabel='Time (s)',ylabel='CWT scale',title='First training window, channel1')
    fig.colorbar(im,ax=ax,label='Absolute coefficient');fig.tight_layout();fig.savefig(out/'cwt_example.png',dpi=150);plt.close(fig)
    distribution=[]
    for split in ['train','test']:
        counts={label:19*sum(trials[i].subject==label for i in plan['holdout'][split]) for label in 'ABCDE'}
        distribution.append({'split':split,**counts,'total':sum(counts.values())})
    write_csv(out/'dataset_summary.csv',distribution)
    return distribution


def error_text(item):
    if not item['maximum_error_pairs']: return item['model']+': seed 42 시험 오분류가 없었다.'
    a,b=item['maximum_error_pairs'][0];example=item['examples'][0];start=int(example['window'])*.15
    return (f"{item['model']}: 최대 오류 {a}→{b} {item['maximum_error_count']}개, 반대 {b}→{a} {item['reverse_error_count']}개였다. "
            f"{b} 예측 {item['destination_predictions']}개(실제190개)이며, 예시 {example['path']}의 {start:.2f}-{start+.3:.2f}초를 오분류했다.")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',required=True)
    parser.add_argument('--data-dir',default=str(ROOT/'data/data'))
    args=parser.parse_args();root=Path(args.run_root).resolve();summary=summarize(root);out=root/'summary'
    distribution=method_figures(out,Path(args.data_dir))
    pdfmetrics.registerFont(TTFont('Nanum',str(report_font())))
    pdfmetrics.registerFont(TTFont('NanumBold',str(report_font(bold=True))))
    body=ParagraphStyle('body',fontName='Nanum',fontSize=9.2,leading=14,spaceAfter=6,wordWrap='CJK')
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=11)
    heading=ParagraphStyle('heading',parent=body,fontName='NanumBold',fontSize=12,leading=17,spaceBefore=8,spaceAfter=6)
    title=ParagraphStyle('title',parent=heading,fontSize=20,leading=27)
    story=[]
    def p(text,style=body): return Paragraph(escape(str(text)),style)
    def add(text,style=body): story.append(p(text,style))
    def table(rows,widths):
        t=Table([[p(v,small) for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EEF2')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        story.extend([t,Spacer(1,6)])
    add('손바닥 sEMG 식별',title);add('딥러닝프로그래밍 Step 1 · 45epoch 비교실험 · 차경호',small)
    add('1 문제 정의',heading)
    add('문손잡이 회전 중 측정한 sEMG로 등록자 A-E를 구분하는 closed-set 5클래스 식별이다. 동일 분할의 Accuracy·macro F1·연산비용으로 DenseNet161과 두 베이스라인을 비교한다. 학습 데이터만 이용하는 BN 재보정도 별도 평가했다.')
    add('2 데이터',heading)
    add('공개 CSV 250개(5명×50시행), 시행당 3초·1,000 Hz·2채널이다. 수치가 동일한 E50을 제외한249시행을 사용자별 비율을 유지해 학습 199 / 시험 50으로 먼저 나눴다(seed 42). 창 생성 후 학습 3,781개·시험 950개이며 시험은 사용자별190개다.')
    table([['윈도우','A','B','C','D','E','합계']]+[[('학습' if row['split']=='train' else '시험')]+[row[k] for k in ['A','B','C','D','E','total']] for row in distribution],[30*mm]+[24*mm]*6)
    add('3 방법',heading)
    add('제공 CSV는 이미 필터링돼 있다. 수업의 60 Hz notch(Q=30)와 4차 20-499 Hz band-pass를 시간축에 추가 적용했다. 각 시행에서300 ms 창·150 ms hop으로19개 창을 만들고 창 전체를 min-max 정규화했다. Morlet(morl) CWT 스케일 1-32의 두 채널 절댓값과 평균으로3×32×300 입력을 만들었다.')
    add('300 ms는 짧은 근활성 패턴과 표본 수를 절충한다. 같은 길이로 모델별 입력 시간 범위를 통제했다. 32스케일은 표현 해상도와 연산량을 절충한다. 수업 설정을 고정해 모델 비교의 변환 조건을 통제했다.')
    add('SimpleCNN·ResNet18·DenseNet161을 모두 사전학습 없이 Adam(lr=0.001), batch 16, Cross-Entropy, 45epoch로 학습했다. 분할은 고정하고 초기화·셔플 seed 42/43/44로3회 반복했으며 마지막45epoch 모델을 평가했다.')
    add('SimpleCNN은 3→32→64의 두 합성곱과 평균 pooling, ResNet18은 잔차 연결과 512→5 분류기, DenseNet161은 조밀 연결과 2208→5 분류기를 쓴다. 공통 학습 설정은 수업 예제를 따라 예산을 통제했다. DenseNet의 첫 7×7 합성곱(stride 2)은 torchvision 기본형이며 추가 dropout은 없어 논문 설명과 차이가 남는다.',small)
    add('DenseNet161+BN은 같은 학습 가중치에서 학습 창만1회 순회해 BN 통계를 재계산한 변형이다(batch 16, 셔플 seed 2026). 별도 개발검증 40시행에서 Accuracy 88.82→89.08%, F1 88.78→89.04%로 선택했다. 차이는760창 중 순증 2개에 불과하며, 시험 점수와 무관하게 원본과 변형을 모두 보고한다.',small)
    figures=Table([[Image(str(out/'filter_check.png'),width=86*mm,height=40*mm),Image(str(out/'cwt_example.png'),width=86*mm,height=40*mm)]],colWidths=[87*mm,87*mm],style=TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
    story.append(figures);add('제공된 필터 신호의 추가 처리 전후와 학습 창의 CWT 예시다.',small);story.append(PageBreak())
    add('4 결과',heading)
    add('시험 50시행의950윈도우를 평가했다. 지표는3회 평균(%)이며 Precision·Recall·F1은 macro 평균이다. 혼동행렬은 seed 42 한 번이며 행=실제, 열=예측, 행 합계=190이다. 빨간 테두리는 최대 오분류다.',small)
    comparison=summary['comparison']
    table([['모델','Accuracy','Precision','Recall','F1']]+[[r['model']]+[pct(r[k+'_mean']) for k in ['accuracy','precision_macro','recall_macro','f1_macro']] for r in comparison],[46*mm,32*mm,32*mm,32*mm,32*mm])
    table([['모델','파라미터','학습/BN 초','추론 ms/창']]+[[r['model'],f"{r['parameters']:,}",f"{r['train_seconds_mean']:.1f} / {r['calibration_seconds_mean']:.1f}",f"{r['inference_ms_per_window_mean']:.2f}"] for r in comparison],[46*mm,42*mm,48*mm,38*mm])
    add(f"평균 Accuracy 최고는 {summary['best_model']}, 최저는 {summary['worst_model']}이다. 동률은 학습+BN 시간이 짧은 쪽을 우선했다.",small)
    images=[Image(str(out/f"confusion_{r['model']}.png"),width=83*mm,height=73*mm) for r in comparison]
    story.append(Table([[images[0],images[1]],[images[2],images[3]]],colWidths=[87*mm,87*mm],style=TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)])))
    story.append(PageBreak())
    add('4 결과 교차검증',heading)
    add('최종 시험 50시행을 제외한199 개발시행 내부에서 stratified5-fold를 수행했다. 각 fold는 새 DenseNet을45epoch 학습했으며 seed 43-47이다. 아래는 원본 / BN 변형이고 SD는 모집단 표준편차다.',small)
    cvs=summary['cv_DenseNet161'];base=cvs['base'];bn=cvs['bn']
    rows=[['Fold','검증 시행','Accuracy % 원본 / BN','Macro F1 % 원본 / BN']]
    for a,b in zip(base['folds'],bn['folds']): rows.append([a['fold'],a['n_validation_trials'],pct(a['accuracy'])+' / '+pct(b['accuracy']),pct(a['f1_macro'])+' / '+pct(b['f1_macro'])])
    rows.append(['평균±SD','199 전체',f"{pct(base['accuracy_mean'])}±{pct(base['accuracy_sd'])} / {pct(bn['accuracy_mean'])}±{pct(bn['accuracy_sd'])}",f"{pct(base['f1_macro_mean'])}±{pct(base['f1_macro_sd'])} / {pct(bn['f1_macro_mean'])}±{pct(bn['f1_macro_sd'])}"])
    table(rows,[22*mm,28*mm,62*mm,62*mm])
    dense=next(row for row in comparison if row['model']=='DenseNet161')
    add(f"논문(Accuracy 94.00%, F1 93.99%, CV Accuracy/F1 91.66/91.64%)과 원본 DenseNet의 관측 차이는 각각 {100*dense['accuracy_mean']-94:+.2f}, {100*dense['f1_macro_mean']-93.99:+.2f}, {100*base['accuracy_mean']-91.66:+.2f}, {100*base['f1_macro_mean']-91.64:+.2f}%p였다. 본 실험의 중복 제거·시행 분할·개발 CV·수업 구현·재사용 시험 조건에서 얻은 차이이며, 특정 원인이나 구조의 우열로 단정하지 않았다.",small)
    add('5 오류 분석',heading)
    for item in summary['errors_seed42']: add(error_text(item),small)
    add('창별 min-max는 각 창의 공통 절대 진폭·오프셋을 제거한다. 이 때문에 비슷한 시간-스케일 모양을 구분하기 어려워지고, 겹치는 창에서 같은 오분류가 반복됐을 가능성이 있다. 이는 전처리 구조에 근거한 가설이며 실제 원인을 확정한 것은 아니다.',small)
    add('6 한계',heading)
    add('1) 5명·단일 세션·실험실 동작 자료다. 미등록자 거절과 다른 날짜·전극 재부착·손잡이 환경은 검증하지 않았다.',small)
    add('2) 같은 시행의 19개 창은 겹치므로950개를 독립 시행으로 볼 수 없다. 최종 50시행은 이전5epoch 결과를 본 뒤 재사용했으므로 탐색적 비교이며 새로운 독립시험이 아니다.',small)
    add('3) 논문의 정확한 분할·CWT·dropout·첫 합성곱 변경 세부가 모두 명시돼 있지 않아94%와 직접적인 우열을 주장하지 않는다. BN 선택과 CV도 같은 개발집합을 사용했으므로 작은 개선의 일반성을 입증하지 않는다.',small)
    add('7 재현 정보',heading)
    add('Linux, Python 3.12.14, PyTorch 2.8.0+cpu, torchvision 0.23.0+cpu를 사용했다(전체 의존성은 requirements.txt). AMD EPYC의 논리 CPU 4개씩 두 작업을 병렬 학습했으며 BN 변형은 원본 학습을 공유한다. 학습시간은 보존된 epoch의 학습 루프만 포함하며 전처리·검증·중단 후 폐기된 계산은 제외한다. 추론은 학습 종료 후 같은 논리 CPU 4개에서 batch 1, 10회 예열 뒤 100회 평균이며 전처리를 제외한다.',small)
    add('설치·실행·파일 역할은 README에 있다. frozen_plan.json은 45epoch 최종 평가 전에 고정한 계획, runs/는 설정·loss·복구 기록, evaluation/은 실제 예측, summary/는 재검산된 지표·그림·보고서다.',small)
    add('제출 저장소: https://github.com/KyunghoCha/semg-auth-coursework · 최종 코드·결과: sEMG_final/',small)
    add('자료: sea3551/palm-sEMG-doorknob-filtered, commit adb7955f4416165c88e4111af6f8fdafd416209c, Data © 2025 Yeonjung Shin, CC BY 4.0. 논문 DOI 10.1038/s41598-026-46294-3. 방법과 베이스라인은 Step 1 2·3주차 강의 예제 기반이다.',small)
    add('생성형 AI 사용: 코드 작성·실험 실행·검증·보고서 정리에 활용했다. 수치는 저장한 실제 예측에서 계산했다.',small)
    def footer(canvas,doc):
        canvas.setFont('Nanum',8);canvas.setFillColor(colors.HexColor('#666666'));canvas.drawRightString(A4[0]-18*mm,9*mm,str(doc.page))
    SimpleDocTemplate(str(out/'report.pdf'),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,bottomMargin=15*mm,title='손바닥 sEMG 식별 45epoch 비교실험',author='차경호').build(story,onFirstPage=footer,onLaterPages=footer)
    write_readme(root,summary,distribution)
    print('Created verified45epoch report and README')


def write_readme(root,summary,distribution):
    report_link=Path(os.path.relpath(root/'summary/report.pdf',ROOT)).as_posix()
    lines=['# 손바닥 sEMG 식별','', f'CWT로 등록자 A-E를 분류하는 Step 1 비교실험이다. 방법·혼동행렬·오류·한계는 [보고서]({report_link})에 있다.','',
      '## 데이터와 방법','', '- 공개250시행 중 동일 신호 E50을 제외한249시행 사용',
      '- 시행을 먼저 학습 199 / 시험 50으로 분할(seed 42), 이후300 ms 창·150 ms hop으로3781/950윈도우 생성',
      '- 이미 필터링된 CSV에 수업의60 Hz notch(Q=30), 4차 20-499 Hz 필터를 추가 적용',
      '- 창 전체 min-max, morl CWT 스케일 1-32, 두 채널 절댓값과 평균: 3×32×300',
      '- 세 모델을 사전학습 없이 Adam 0.001·batch 16·45epoch로 seed 42/43/44 각각 학습',
      '- DenseNet+BN은 같은 가중치에 학습 데이터만 이용한 BN 재보정 1회 추가. 개발검증에서 순증2/760창의 작은 차이로 선택',
      '- 모델 구조: SimpleCNN은 두 합성곱(3→32→64)과 평균 pooling, ResNet18은 잔차 연결과 512→5 분류기, DenseNet161은 조밀 연결과2208→5 분류기',
      '- DenseNet은 torchvision 기본 첫 합성곱과 dropout 없음. 공통 학습 설정은 수업 예제를 따랐으며 논문 구현의 미공개 세부까지 일치한다고 주장하지 않는다',
      '- 최종 시험 50시행은 이전5epoch 결과를 본 뒤 재사용했다. DenseNet5-fold는199 개발시행 내부에서만 수행','',
      '분할은 [data_utils.py](data_utils.py)의 split_trials와 experiment.py의 make_plan에서 시행 단위로 먼저 수행한다. 창 생성은 그 이후다.','',
      '| 윈도우 | A | B | C | D | E | 합계 |','|---|---:|---:|---:|---:|---:|---:|',
      *['| '+('학습' if row['split']=='train' else '시험')+' | '+' | '.join(str(row[k]) for k in ['A','B','C','D','E','total'])+' |' for row in distribution],'',
      '## 실제 결과','', '950윈도우(원본 50시행)의3회 평균(%). Precision·Recall·F1은 macro 평균이다.','',
      '| 모델 | Accuracy | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for r in summary['comparison']: lines.append('| '+r['model']+' | '+' | '.join(pct(r[k+'_mean']) for k in ['accuracy','precision_macro','recall_macro','f1_macro'])+' |')
    lines += ['',f"최고: {summary['best_model']}, 최저: {summary['worst_model']}(평균 Accuracy 기준). 모델별 혼동행렬과 오류, fold별 값·평균±SD, 시간·파라미터는 보고서에 있다.",'',
      '## 실행','', 'Linux, Python 3.12, 가용 논리 CPU 최소 4개, flock/taskset이 필요하다. 아래 기본 명령은 논리 CPU 4개 1작업이다. 실제 기록은 가용 논리 CPU 9개에서 각 4개를 쓰는 두 작업을 병렬 실행했다. 전체 재학습에는 수 시간이 걸린다.','',
      'PDF 재생성에는 NanumGothic 글꼴이 필요하다. Ubuntu: sudo apt-get install fonts-nanum. 다른 환경은 assets/README.md를 참고한다.','',
      '```bash','python -m venv .venv','source .venv/bin/activate',
      'python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu',
      'python -m pip install -r requirements.txt','git clone https://github.com/sea3551/palm-sEMG-doorknob-filtered.git data',
      'git -C data checkout adb7955f4416165c88e4111af6f8fdafd416209c','python -m unittest -v test_project.py test_summary45.py',
      'mkdir -p rerun45','cp results45/frozen_plan.json rerun45/',
      'python run_cpu45.py --output-root rerun45 --cache-dir cache45 --workers 1',
      'python complete_evaluation45.py --run-root rerun45 --cache-dir cache45 --evaluate-test',
      'python make_report45.py --run-root rerun45','```','',
      '완료 체크포인트는 재사용한다. 새 실험은 별도 출력 폴더와 사전 계획을 사용하며 시험 결과로 설정을 바꾸지 않는다. 재학습 없이 수치를 재검산하려면 python summarize45.py --run-root results45를 실행한다.','',
      'BN 선택검증 재현(선택):','', '```bash',
      'python colab_run.py --job validate --model DenseNet161 --seed 42 --epochs 45 --device cpu --threads 4 --data-dir data/data --cache-dir cache45 --output-dir rerun_validation',
      'python bn_recalibrate.py --run-dir rerun_validation --output-dir rerun_bn --data-dir data/data --cache-dir cache45 --threads 4',
      '```','', '## 파일','', '| 파일 | 역할 |','|---|---|',
      '| data_utils.py | CSV 검증·중복 제거·시행 단위 분할 |',
      '| experiment.py | 필터·CWT·세 모델·결정론 설정의 공통 함수, 5epoch 기준실험 |',
      '| colab_run.py | 단일45epoch 작업, 모델·Adam·RNG 체크포인트 복구 |',
      '| run_cpu45.py | 고정된3모델×3seed와 DenseNet5-fold 학습; 시험 평가는 차단 |',
      '| finalize45.py / complete_evaluation45.py | 고정 계획 검사, 순차 평가, DenseNet BN 변형 |',
      '| summarize.py | 혼동행렬 그림 공통 함수 |',
      '| summarize45.py / make_report45.py | 예측 재검산, 표·혼동행렬·PDF·README |',
      '| bn_recalibrate.py | 개발검증에서 수행한 BN 후보 진단과 반복 일치 검사 |',
      '| test_project.py / test_summary45.py | 중복·누수·분할·CWT·모델 출력·예측 파일 검사 |',
      '| results45/ | 고정 계획·실제 loss·설정·예측·지표·보고서 |',
      '| results45/development/ | BN 선택 당시의 비교 지표와 원본·재보정 예측 |',
      '| assets/ | PDF 글꼴 설치 안내와 NanumGothic OFL 라이선스 |','',
      '데이터·CWT 캐시·대형 가중치는 Git에서 제외한다. 재학습은 공개 데이터를 내려받아 재생성한다.','',
      '## 한계와 출처','',
      '5명·단일 세션의 closed-set 결과이며 겹치는 창은 독립 시행이 아니다. 재사용 시험집합과 개발집합에서 선택한 BN 변형의 결과를 새로운 독립시험이나 논문94%의 정확한 재현으로 주장하지 않는다.','',
      '- 데이터: https://github.com/sea3551/palm-sEMG-doorknob-filtered · Data © 2025 Yeonjung Shin, CC BY 4.0',
      '- 논문: https://doi.org/10.1038/s41598-026-46294-3',
      '- BN 통계 갱신: https://docs.pytorch.org/docs/2.8/optim.html#torch.optim.swa_utils.update_bn',
      '- 코드 구성: Step 1 2·3주차 강의 예제 기반',
      '- 생성형 AI 사용: 코드 작성·실험·검증·보고서 정리. 결과는 실제 실행과 예측에서 계산','']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
