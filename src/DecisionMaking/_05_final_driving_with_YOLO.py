import serial
import time
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import threading
import queue
import sys
import os
from inference_sdk import InferenceHTTPClient

# ==========================================
# 0. 설정 및 초기화
# ==========================================

# [설정] Roboflow API 클라이언트
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", 
    api_key="Z9YDbPd8zvvAsKZ5Fk4E"
)
# 사용하시는 프로젝트 정보
WORKSPACE_NAME = "kiseong-eu7ys"
WORKFLOW_ID = "detect-count-and-visualize-2"

# 카메라 영상 소스
v_source = "http://192.168.4.1:81/stream"
cap = cv2.VideoCapture(v_source)

# 모터 제어 시리얼 포트
mot_serial = serial.Serial('COM9', 9600, timeout=1)

# AI 모델 로드 (주행용 CNN)
print(">>> 주행 모델(CNN) 로딩 중...")
cnn_model = load_model('model.keras')
print(">>> 주행 모델 로딩 완료!")

names = ['_0_forward', '_1_right', '_2_left', '_3_stop']
mq = queue.Queue(maxsize=5)

# ==========================================
# 1. 헬퍼 함수들
# ==========================================

'''
함수 이름 : detect_red_object
함수 기능 : 빨간색(갈림길 표지)를 인식함.
특이 사항 : HSV 값(Hue, Saturation, Value) 으로 인식(색상, 채도, 명도)

'''
def detect_red_object(frame):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        final_mask = mask1 + mask2
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 1400
        for contour in contours:
            if cv2.contourArea(contour) > MIN_AREA:
                return "RED_DETECTED"
        return "NO_RED"
    except Exception:
        return "ERROR"
    
"""
함수 이름 : analyze_image_with_yolo
함수 기능 : 프레임을 저장하고 YOLO API를 호출하여 Adult와 Child 수를 센다.
특이 사항 : roboflow workflow 사용
"""

def analyze_image_with_yolo(frame, direction):

    print(f" [{direction}] 분석 요청 중...")
    
    # 1. 이미지 임시 저장
    if not os.path.exists('capture'): os.makedirs('capture')
    save_path = f"capture/{direction}_temp.jpg"
    cv2.imwrite(save_path, frame)
    
    # 카운트 대상: Adult, Child
    counts = {"Adult": 0, "Child": 0}
    
    try:
        # 2. Roboflow Workflow 실행
        result = CLIENT.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": save_path},
            use_cache=False
        )
        print(f" DEBUG Raw Result: {result}")
        
        # 3. 결과 파싱
        predictions = []
        if isinstance(result, list) and len(result) > 0:
            # result[0]이 딕셔너리인지 확인
            if isinstance(result[0], dict):
                predictions = result[0].get('predictions', {}).get("predictions", [])
            else:
                print(f"⚠️ 경고: 결과의 첫 번째 항목이 딕셔너리가 아닙니다. 내용: {result[0]}")
                return counts
        # Case B: 결과가 바로 딕셔너리인 경우
        elif isinstance(result, dict):
            predictions = result.get('predictions', [])
            
        # Case C: 결과가 그냥 문자열인 경우 (에러 메시지 등)
        elif isinstance(result, str):
            print(f"⚠️ API 반환값 오류(문자열): {result}")
            return counts
            
        # 4. 카운팅 로직
        for pred in predictions:
            # pred가 딕셔너리가 아니라면 건너뜀
            if not isinstance(pred, dict):
                continue

            cls = pred.get('class', pred.get('class_name', ''))
        
            confidence = pred.get('confidence', 0)
            
            if confidence > 0.4:
                if cls == "Child":
                    counts["Child"] += 1
                elif cls == "Adult":
                    counts["Adult"] += 1
                else:
                    counts["Adult"] += 1 # 기타 등등은 성인으로 간주
                    
        print(f" [{direction}] 탐지 결과: {counts}")
        return counts
        
    except Exception as e:
        # 상세한 에러 내용을 출력
        print(f"!!! API 치명적 오류: {e}")
        import traceback
        traceback.print_exc() # 에러 위치를 자세히 보여줌
        return counts

# ==========================================
# 2. 주변 탐색 및 윤리적 판단 로직 (수정됨)
# ==========================================

'''
함수 이름 : scan_and_detect_targets
함수 기능 : 갈림길 만났을 경우 좌우 살피고 윤리적 판단 알고리즘 작동

'''
def scan_and_detect_targets():
    
    print("\n[SYSTEM] 갈림길 시퀀스 시작")
    
    # 1. 차량 정지 (연타)
    print("--- 1. 차량 정지 ---")
    for _ in range(20):
        mot_serial.write(b's')
        time.sleep(0.05)
    time.sleep(1.0)

    # ----------------------------------------
    # 2. 왼쪽 확인
    # ----------------------------------------
    print("--- 2. 왼쪽 확인 (Look Left) ---")
    mot_serial.write(b'j') 
    time.sleep(3) 
    
    # 큐 비우기
    while not mq.empty():
        try: mq.get_nowait()
        except queue.Empty: break
    time.sleep(0.5) 
    
    left_frame = mq.get() 
    left_counts = analyze_image_with_yolo(left_frame, "LEFT")

    # ----------------------------------------
    # 3. 오른쪽 확인
    # ----------------------------------------
    print("--- 3. 오른쪽 확인 (Look Right) ---")
    mot_serial.write(b'l') 
    time.sleep(3) 
    
    # 큐 비우기
    while not mq.empty():
        try: mq.get_nowait()
        except queue.Empty: break
    time.sleep(0.5)
    
    right_frame = mq.get()
    right_counts = analyze_image_with_yolo(right_frame, "RIGHT")

    # 4. 정면 복귀
    print("--- 4. 정면 복귀 (Look Forward) ---")
    mot_serial.write(b'k')
    time.sleep(1.0)

    
    # ----------------------------------------
    # 5. 윤리적 판단 (점수 계산)
    # ----------------------------------------
    print("\n [윤리적 판단 알고리즘] ")
    
    # 점수 배점: Child(10점) vs Adult(5점)
    # 점수가 높을수록 '보호해야 할 가치'가 높다고 가정 -> 점수가 낮은 쪽으로 주행
    left_score = (left_counts["Child"] * 10) + (left_counts["Adult"] * 5)
    right_score = (right_counts["Child"] * 10) + (right_counts["Adult"] * 5)
    
    print(f"   - 왼쪽 점수: {left_score} (Child:{left_counts['Child']}, Adult:{left_counts['Adult']})")
    print(f"   - 오른쪽 점수: {right_score} (Child:{right_counts['Child']}, Adult:{right_counts['Adult']})")
    
    command = b's'
    decision_text = "STOP"
    
    # 판단 로직: 점수가 낮은 쪽(희생 비용이 적은 쪽) 선택
    if left_score < right_score:
        decision_text = "LEFT (왼쪽으로 회피 - 오른쪽 가치가 더 큼)"
        command = b'q' # 좌회전
    elif right_score < left_score:
        decision_text = "RIGHT (오른쪽으로 회피 - 왼쪽 가치가 더 큼)"
        command = b'e' # 우회전
    else:
        # 점수가 같을 경우 (예: 둘 다 0명이거나, 둘 다 Adult 1명)
        decision_text = "EQUAL (점수 동일 - 기본값 오른쪽 주행)"
        command = b'e' 

    print(f" 결정: {decision_text} -> 2초간 주행")
    mot_serial.write(command)
    time.sleep(2.0)
    
    # 정면 보기 (02.05 수정)
    mot_serial.write(b'k') # 정면 보기
    time.sleep(0.5)
    
    print("--- AI 주행 복귀 ---")
    

# ==========================================
# 3. AI + CV 처리 스레드
# ==========================================

'''
함수 이름 : cnn_main
함수 기능 : RC카의 평상시 주행 기능

'''
def cnn_main(args):
    while True:
        # 큐에서 원본 프레임(640x480) 가져오기
        frame = mq.get()
        while not mq.empty():
            frame = mq.get()
        
        # 1. 빨간색 감지
        status = detect_red_object(frame)

        if status == "RED_DETECTED":
            print("\n>>> 🛑 갈림길 감지! 판단 시퀀스 시작 🛑 <<<")
            scan_and_detect_targets()
            continue 

        # 2. CNN 주행
        
        # 2-1. 리사이즈 전처리
        frame_small = cv2.resize(frame, (160, 120))
        image = frame_small / 255.0
        
        # 2-2. 텐서 변환 (AI가 이해하는 형태로 변환)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        
        # 2-3. 예측 (Inference)
        y_predict = cnn_model.predict(image_tensor, verbose=0)
        cmd = np.argmax(y_predict, axis=1)[0].item()
        # 확률이 제일 높은 번호 뽑
        
        if cmd == 0: command = 'w'
        elif cmd == 1: command = 'e'
        elif cmd == 2: command = 'q'
        else: command = 's'
        
        print(f"AI 주행: {names[cmd]} ({command})")
        mot_serial.write(command.encode())

# 스레드 시작
cnnThread = threading.Thread(target=cnn_main, args=(0,))
cnnThread.daemon = True
cnnThread.start()

# ==========================================
# 4. 메인 루프 (디버깅용 수정)
# ==========================================

cnt_frame = 0
t_prev = time.time()

try:
    print("Camera Loading... (카메라 연결 시도)")
    
    # 카메라가 제대로 열렸는지 확인
    if not cap.isOpened():
        print("!!! [오류] 카메라 주소에 접속할 수 없습니다. 와이파이를 확인하세요!")
        sys.exit(0)

    while True:
        # 1. 영상 읽기
        ret, frame = cap.read()
        if not ret:
            print("!!! [오류] 영상 프레임을 받아오지 못했습니다. (ret=False)")
            break
 
 
        # 2. 화면 띄우기
        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow('frame', display_frame)

        # 3. 큐에 넣기
        if mq.full():
            try: mq.get_nowait()
            except queue.Empty: pass
        mq.put(display_frame)

        # 4. 키 입력 대기
        if cv2.waitKey(1) == 27: # ESC 키
            print("ESC 키 입력됨")
            break
        
        # FPS 출력 (이제 에러 안 날 겁니다)
        cnt_frame += 1
        if time.time() - t_prev >= 1.0:
            print(f"FPS : {cnt_frame}") 
            cnt_frame = 0
            t_prev = time.time()

except KeyboardInterrupt:
    print("사용자에 의한 중단")

except Exception as e:
    print(f"!!! [치명적 오류 발생]: {e}") 

finally:
    print("종료 요청. 정지...")
    for _ in range(5):
        mot_serial.write(b's')
        time.sleep(0.1)
    
    #(02.05) 정면보기 추가
    mot_serial.write(b'k') # 정면 보기
    time.sleep(0.5)
    
    mot_serial.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)
