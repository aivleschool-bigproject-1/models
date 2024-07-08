## 키포인트 연결해서 사람 보여주는 코드
import cv2
from ultralytics import YOLO

# 웹캡 캡처 설정
cap = cv2.VideoCapture(0)

# YOLO 모델 로드 (YOLOv8 모델)
model = YOLO("yolov8n-pose.pt")

# Define connections based on the uploaded image
connections = [
    (0, 1), (0, 2), (2, 4), (1, 3),  # Head
    (5, 6), (6, 8), (8, 10),  # Right arm
    (5, 7), (7, 9),  # Left arm
    (5, 11), (11, 13), (13, 15),  # Right leg
    (6,12),(11, 12), (12, 14), (14, 16)  # Left leg
]

# 웹캠에서 프레임을 캡처하고 처리하는 루프
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
        keypoints = result.keypoints.xy.cpu().numpy()  # Keypoints

        # 각 객체마다 박스 그리기
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(result.boxes.cls[i])  # 클래스 ID 얻기
            label = model.names[cls_id]  # Object name 얻기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # 각 객체마다 포즈 키포인트 그리기 (키포인트가 박스 내에 있을 때만)
            person_keypoints = keypoints[i]
            points_in_box = []
            for point in person_keypoints:
                x, y = map(int, point)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    points_in_box.append((x, y))
            
            # 키포인트 간 연결 그리기 (둘 다 박스 내에 있을 때만)
            for connection in connections:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(person_keypoints) and pt2_idx < len(person_keypoints):
                    pt1 = person_keypoints[pt1_idx]
                    pt2 = person_keypoints[pt2_idx]
                    x1, y1 = map(int, pt1)
                    x2, y2 = map(int, pt2)
                    if (x1, y1) in points_in_box and (x2, y2) in points_in_box:
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # 결과 프레임을 디스플레이
    cv2.imshow('YOLOv8 Pose Detection', frame)
    
    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
