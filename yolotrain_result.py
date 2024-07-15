## 학습한 모델 이용
import cv2
import os
import pandas as pd
from ultralytics import YOLO
import time

# 웹캡 캡처 설정
video_path = 'fallpeople.mp4'
cap = cv2.VideoCapture(0)

# YOLO 모델 로드 (YOLOv8 모델)
model = YOLO("v8_nano_results.pt", verbose=False)
#v3 : 2024-07-06apply all augumentation to imbalanced dataset(number of dataset 805->19,11) 
#v4 : 2024-07-09,apply all augumentation to more balanced than v3 (number of dataset 1041 -> 2477) 성능 좋았음
#v5 : 2024-07-09,not augumentation (number of dataset 1041) small better chatched fallperson than nano, nano catched better than small others
#v6 : 2024-07-10, apply all augumentation * 5, v6.4 : init labels and label again to 1022 images(성능 좋았음), v6.5 : augumentation * 5, v6.6 : augumentation *3
# 웹캠에서 프레임을 캡처하고 처리하는 루프
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        cv2.waitKey()
        break

    # YOLO 모델을 사용하여 포즈 감지 수행
    results = model(frame)
    # 감지된 객체 및 포즈 시각화
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        
        for i, box in enumerate(boxes):
            if scores[i] >= 0.6:
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(result.boxes.cls[i])  # 클래스 ID 얻기
                label = model.names[cls_id]  # Object name 얻기
                score = scores[i]  # confidence score 얻기
                label_with_score = f"{label} {score:.2f}"
                
                if label == 'Person':
                    box_color = (0, 255, 0)
                    text_color = (36, 255, 12)
                elif label == 'Fallperson':
                    box_color = (224, 0, 0)
                    text_color = (224, 0, 0)   
                elif label == 'Safehat':
                    box_color = (255, 255, 255)
                    text_color = (255, 255, 255)
                else : 
                    box_color = (0,0,255)
                    text_color = (0, 0, 255)
        
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label_with_score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                person_keypoints = keypoints[i]
                for point in person_keypoints:
                    x, y = map(int, point)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                
    # 결과 프레임을 디스플레이
    cv2.imshow('YOLOv8 Pose Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
end_time = time.time()
total = (end_time-start_time)
print(total)
# 리소스 해제
cap.release()
cv2.destroyAllWindows()