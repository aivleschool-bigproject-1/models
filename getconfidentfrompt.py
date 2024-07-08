## 키포인트 안보여주고 신뢰도만 보여주는 코드 
import cv2
from ultralytics import YOLO

# 웹캠 캡처 설정
cap = cv2.VideoCapture(0)

# YOLO 모델 로드
model = YOLO("../models/yolov8n-pose.pt")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        break

    # YOLO 모델로 객체 인식 수행
    results = model(frame)
    
    # 결과에서 인식된 객체들의 경계 상자 정보 추출
    for result in results:
        boxes = result.boxes  # 경계 상자들
        
        for box in boxes:
            # 경계 상자 좌표 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표를 정수형으로 변환
            confidence = box.conf[0]  # 신뢰도
            class_id = int(box.cls[0])  # 클래스 ID
            
            # 경계 상자 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 클래스 이름 및 신뢰도 표시
            label = f"{model.names[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 결과 화면 출력
    cv2.imshow("YOLO Object Detection", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
