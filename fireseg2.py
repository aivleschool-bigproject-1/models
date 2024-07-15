# 바운딩 박스와, 객체에 마스크 쳐지는 세그멘테이션
import cv2
import os
import numpy as np
from ultralytics import YOLO

# 비디오 파일 경로
video_path = 'fire.mp4'
cap = cv2.VideoCapture(video_path)

# YOLO 모델 로드 (YOLOv8 모델)
model = YOLO("fire_seg_results.pt", verbose=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델을 사용하여 객체 감지 및 세그멘테이션 수행
    results = model(frame)

    # 감지된 객체 및 세그멘테이션 시각화
    for result in results:
        if result.masks :
            masks = result.masks.xy
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            if scores[i] >= 0.3:
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(result.boxes.cls[i])  # 클래스 ID 얻기
                label = model.names[cls_id]  # Object name 얻기
                score = scores[i]  # confidence score 얻기
                label_with_score = f"{label} {score:.2f}"
                
                # 경계 상자 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label_with_score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # 마스크 그리기
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    for m in mask:
                        m = m.astype(int)
                        frame[m[1], m[0]] = [255, 0, 255]

    # 결과 프레임을 디스플레이
    cv2.imshow('YOLOv8 Segmentation', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
