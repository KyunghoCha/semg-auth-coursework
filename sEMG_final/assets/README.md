# PDF 글꼴

저장된 PDF는 별도 글꼴 설치 없이 읽을 수 있다. PDF를 다시 만들 때는 NanumGothic 일반체·굵은체가 필요하다.

Ubuntu에서는 `sudo apt-get install fonts-nanum`으로 설치한다. 다른 환경에서는 NanumGothic 글꼴 파일을 준비한 뒤 `SEMG_FONT_DIR`을 해당 폴더로 지정한다. 지원 파일명은 `NanumGothic-Regular.ttf` / `NanumGothic-Bold.ttf` 또는 `NanumGothic.ttf` / `NanumGothicBold.ttf`다. 이 assets 폴더에 두어도 자동으로 찾는다.

글꼴 바이너리는 저장소에 포함하지 않았다. NanumGothic의 SIL Open Font License는 [OFL.txt](OFL.txt)에 보존했다.
