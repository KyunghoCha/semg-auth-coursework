import numpy as np, glob, os

files = sorted(glob.glob('data/**/*.csv', recursive=True))
print('파일 개수:', len(files))
print('예시 경로:', files[0])

x = np.loadtxt(files[0], delimiter=',', skiprows=1) # 형식에 맞게 변경
print('배열 모양:', x.shape) # 예: (3000, 2)
print('값 범위:', x.min(), '~', x.max())

# 피험자별 파일 수 세기
from collections import Counter
subs = [os.path.basename(os.path.dirname(f)) for f in files]
print(Counter(subs))
