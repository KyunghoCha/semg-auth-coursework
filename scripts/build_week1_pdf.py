"""Render the executed notebook's evidence to an A4 PDF (ReportLab)."""
from pathlib import Path
import json, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
a=json.loads((OUT/"analysis.json").read_text(encoding="utf-8"))
env=json.loads((OUT/"environment.json").read_text(encoding="utf-8"))
# Image of captured stdout, without a simulated terminal UI.
from PIL import Image as PILImage, ImageDraw, ImageFont
mono_candidates=[Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),Path("C:/Windows/Fonts/consola.ttf")]
mono=ImageFont.truetype(str(next(f for f in mono_candidates if f.exists())),20)
lines=["ENVIRONMENT CHECK / ACTUAL STDOUT", "Executed: "+env["executed_at"], ""]+env["stdout"].splitlines()
im=PILImage.new("RGB",(1360,52+len(lines)*30),(247,250,252))
draw=ImageDraw.Draw(im)
for i,line in enumerate(lines):
    draw.text((32,22+i*30),line,font=mono,fill=(20,50,74))
im.save(OUT/"images/environment_check.png")
font_candidates=[
 (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"), Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")),
 (Path("C:/Windows/Fonts/malgun.ttf"),Path("C:/Windows/Fonts/malgunbd.ttf"))]
regular,bold=next((r,b) for r,b in font_candidates if r.exists() and b.exists())
pdfmetrics.registerFont(TTFont("Korean",str(regular)))
pdfmetrics.registerFont(TTFont("KoreanBold",str(bold)))
pdfmetrics.registerFontFamily("Korean",normal="Korean",bold="KoreanBold")
NAVY=colors.HexColor("#14324a")
TEAL=colors.HexColor("#146b82")
GRAY=colors.HexColor("#536474")
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name="KBody",fontName="Korean",fontSize=9.5,leading=15,wordWrap="CJK",spaceAfter=7,textColor=NAVY))
styles.add(ParagraphStyle(name="KSmall",parent=styles["KBody"],fontSize=8,leading=12,spaceAfter=5,textColor=GRAY))
styles.add(ParagraphStyle(name="KTitle",parent=styles["KBody"],fontName="KoreanBold",fontSize=23,leading=32,spaceAfter=12))
styles.add(ParagraphStyle(name="KHead",parent=styles["KBody"],fontName="KoreanBold",fontSize=14,leading=20,spaceBefore=8,spaceAfter=8))
styles.add(ParagraphStyle(name="KSub",parent=styles["KBody"],fontName="KoreanBold",fontSize=10.5,leading=16,spaceBefore=6,spaceAfter=5))
styles.add(ParagraphStyle(name="KCell",parent=styles["KBody"],fontSize=8,leading=12,spaceAfter=0))
story=[]
W=174*mm
def p(text,style="KBody"):
    return Paragraph(html.escape(str(text)).replace("\n","<br/>"),styles[style])
def add(text,style="KBody"): story.append(p(text,style))
def head(text): add(text,"KHead")
def table(rows,widths,last_emphasis=False):
    vals=[[p(v,"KCell") for v in row] for row in rows]
    t=Table(vals,colWidths=[v*mm for v in widths],repeatRows=1,hAlign="LEFT")
    rules=[("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e4edf3")),
           ("VALIGN",(0,0),(-1,-1),"TOP"),
           ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#96aeba")),
           ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#d5dee4")),
           ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
           ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]
    if last_emphasis: rules.append(("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#eaf4ef")))
    t.setStyle(TableStyle(rules));story.append(t);story.append(Spacer(1,6*mm))
def fig(subject):
    image=Image(str(OUT/"images"/f"subject_{subject}.png"),width=W,height=W*4.7/10)
    story.append(image)
    add(f"그림 {'ABCDE'.index(subject)+1}. 피험자 {subject}의 1번 시행. 채널 1(APB)과 채널 2(ADM)를 시간축으로 표시했다.","KSmall")
def newpage():story.append(PageBreak())

add("딥러닝프로그래밍 | 1주차 실습","KSmall")
add("손바닥 sEMG 데이터 탐색\n및 파형 비교","KTitle")
add("공개 데이터의 구조를 점검하고 피험자별 근전도 파형을 비교하였다. 등록된 사용자 5명을 구분하는 문제를 이해하기 위한 탐색 단계이며, 이번 실습에서는 분류 모델을 학습하지 않았다.")
head("1. 환경 확인")
add("WSL · Anaconda semg · Python "+env["python"]+" · RTX 3070 Laptop GPU","KSub")
image=OUT/"images/environment_check.png"
if image.exists():
    from PIL import Image as PILImage
    with PILImage.open(image) as im: iw,ih=im.size
    story.append(Image(str(image),width=W,height=W*ih/iw))
else:
    for line in env["stdout"].splitlines():add(line,"KSmall")
add("그림 0. 실제 Python 점검 명령의 표준출력을 이미지로 저장한 결과. 패키지 import, GPU 인식, GPU 텐서 연산을 확인했다.","KSmall")
head("2. 데이터 요약")
rows=[["피험자","파일/시행 수","파일별 배열 크기","시행 시간","최솟값 / 최댓값"]]
for s in a["summary"]:
    rows.append([s["subject"],s["trials"],s["shape"],"3초",f'{s["minimum"]:.5f} / {s["maximum"]:.5f}'])
table(rows,[16,26,43,22,67])
add("전체 250개 파일 모두 (3000, 2), 1,000 Hz, 두 채널이며 NaN·무한대가 없었다. 피험자마다 파일명상 시행 번호 1–50이 존재했다.","KSmall")
add("자료 점검: E의 38번·50번 파일은 바이트 내용이 같아 고유 파일 내용은 249개였다. 원자료는 유지했으며 독립 시행 여부는 이 파일만으로 확인할 수 없다.","KSmall")

newpage()
head("3. 피험자별 파형 — A와 B")
add("피험자별 파일명상 1번 시행을 같은 기준으로 선택했다. 배경 영역과 점선은 파지(0–1초), 회전(1–2초), 정지(2–3초)를 나타낸다. 진폭은 단위 환산 없이 CSV 기록값을 사용했다.","KSmall")
fig("A")
story.append(Spacer(1,3*mm))
fig("B")
add("각 패널의 y축 범위는 자동 설정하였다. 파형의 높이를 비교할 때 축 눈금을 함께 읽어야 한다.","KSmall")

newpage()
head("3. 피험자별 파형 — C와 D")
add("동작 경계는 데이터 수집 프로토콜에 따른 시간 구분이다. 신호에서 실제 동작 시작점을 검출해 정렬한 결과가 아니다.","KSmall")
fig("C")
story.append(Spacer(1,3*mm))
fig("D")
add("채널 1은 짧은엄지벌림근(APB), 채널 2는 새끼벌림근(ADM)에 대응한다. 두 채널의 원래 열 이름은 각각 Comp Ch 3, Comp Ch 4이다.","KSmall")

newpage()
head("3. 피험자별 파형 — E")
fig("E")
head("4. 관찰을 뒷받침하는 구간별 RMS")
add("RMS = sqrt(mean(x²)). 각 1초 구간의 1,000개 값으로 계산했으며 단위는 CSV 기록값과 같다. 아래 표는 각 피험자의 1번 시행에만 해당한다.","KSmall")
rows=[["피험자","채널","파지 RMS","회전 RMS","정지 RMS"]]
for v in a["phase_rms"]:
    rows.append([v["subject"],v["channel"],f'{v["grasp"]:.4f}',f'{v["rotate"]:.4f}',f'{v["stop"]:.4f}'])
table(rows,[20,20,44,45,45])
add("자료의 물리 단위를 README만으로 확정할 수 없어 V 또는 mV로 표기하지 않았다. 이미 필터링된 공개 신호에 추가 필터링·정규화를 적용하지 않았다.","KSmall")

newpage()
head("5. 관찰 메모 세 가지")
for i,ob in enumerate(a["observations"],1):
    add(f"관찰 {i}","KSub");add(ob)
add("해석 범위: 총 5개 대표 시행의 관찰이다. 피험자별 50회 전체의 특성이나 날짜가 달라졌을 때의 재현성, 사용자 식별 정확도로 일반화하지 않는다.","KSmall")
head("6. 선행연구 비교")
add("원논문 Table 1의 다섯 연구를 재구성하고 마지막 행에 대상 논문을 추가했다. [17]–[21]은 원논문의 참고문헌 번호이다.","KSmall")
table([a["related_columns"]]+a["related_rows"],[28,19,12,12,31,44,28],True)
add("정확도는 각 논문에서 보고된 값이며 서로 다른 평가 조건의 수치다. 대상 논문의 94.00%는 Table 3의 시험 정확도이다.","KSmall")

newpage()
head("7. 선행연구와의 차이점")
for i,line in enumerate(a["differences"],1):add(f"{i}. {line}")
head("8. 출처 및 재현 방법")
add("원논문","KSub")
add("Shin, Y., Kim, J., & Choi, S.-I. (2026). Palm sEMG-based user identification during doorknob rotation using a convolutional neural network. Scientific Reports, 16, 22244.","KSmall")
add("https://doi.org/10.1038/s41598-026-46294-3","KSmall")
add("Table 1: 선행연구의 측정 조건·정확도. Table 3: 대상 논문의 정확도. 선행연구 표는 이 원논문을 통해 인용했으며 각 선행연구를 별도로 재현하지 않았다.","KSmall")
add("공개 데이터","KSub")
add(a["data_repository"],"KSmall")
add("사용한 커밋: "+a["data_commit"],"KSmall")
add("Data © 2025 Yeonjung Shin, CC BY 4.0. README에 따라 60 Hz 노치와 20–500 Hz 대역통과 필터가 이미 적용된 데이터를 사용했다. 250개 원본 파일의 SHA-256은 data_manifest.csv에 기록했다.","KSmall")
add("강의자료","KSub")
add("Step1_1주차_강의와실습.pdf — 14쪽: 선행연구 표 재작성, 17쪽: 환경 확인·데이터 요약·3명 이상 파형·관찰 3개·비교표 제출 요건.","KSmall")
add("실행 및 결과 파일","KSub")
add("저장소: "+a["repository_url"],"KSmall")
add("week1_report.ipynb를 semg 커널에서 처음부터 실행하면 구조 검사, 파형, RMS, 표와 관찰 문장을 재생성한다. 의존성·데이터 받기·명령 실행 절차는 저장소 README에 기록했다.","KSmall")
add("실행 시각: "+a["executed_at"]+" (Asia/Seoul)","KSmall")
head("범위와 용어")
add("파일 하나는 한 피험자의 3초 시행이며 두 채널을 함께 포함한다. 이번 결과는 환경과 데이터 탐색에 해당한다. 논문의 센서는 실제 실험에서 손바닥에 부착되었으며, 문손잡이에 통합된 완제품의 검증 결과로 해석하지 않는다.","KSmall")
add("DWT: 이산 웨이블릿 변환 · EWT: 경험적 웨이블릿 변환 · EMD: 경험적 모드 분해 · CQT: 상수-Q 변환 · CWT: 연속 웨이블릿 변환 · DNN: 심층 신경망 · CNN: 합성곱 신경망.","KSmall")
add("Table 1의 GAN은 원문의 열 구성을 유지했으며 데이터 증강에 사용된 방법이다. Lu 등의 최대 정확도는 원문의 99.206%를 유지했다(강의자료에서는 약 99.21%).","KSmall")

def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d5dee4"))
    canvas.line(18*mm,16*mm,A4[0]-18*mm,16*mm)
    canvas.setFont("Korean",8);canvas.setFillColor(GRAY)
    canvas.drawString(18*mm,11*mm,"손바닥 sEMG | 1주차 데이터 탐색")
    canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))
    canvas.restoreState()
target=OUT/"week1_report.pdf"
doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
                     topMargin=16*mm,bottomMargin=22*mm,
                     title="손바닥 sEMG 데이터 탐색 및 파형 비교",
                     author="KyunghoCha",subject="딥러닝프로그래밍 1주차 실습")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(target)
