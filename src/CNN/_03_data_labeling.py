import os
import csv

dataDir='data.1695035538.287210' # 데이터 저장 디렉터리

print(os.getcwd())  # 현재 디렉터리 어딘지 확인
os.chdir(dataDir)   # 디렉터리 이동
roadDirs = os.listdir() # 현재 디렉터리 확인
print(roadDirs)

f_csv = open('0_road_labels.csv','w', newline='')
wr = csv.writer(f_csv)
print(wr.writerow(["file","label", "labelNames"]))

roadDirs = [road for road in roadDirs if os.path.isdir(road)]
print(roadDirs)

for num, roadDir in enumerate(roadDirs):
    roadFiles=os.listdir(roadDir)
    for roadFile in roadFiles:
        wr.writerow([os.path.join(roadDir, roadFile),num,roadDir])

f_csv.flush()
f_csv.close()

