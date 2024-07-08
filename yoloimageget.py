## 사진도 찍고 데이터로도 만들고 
import cv2
import os
import pandas as pd
from ultralytics import YOLO

# 웹캡 캡처 설정
cap = cv2.VideoCapture(1)

# YOLO 모델 로드 (YOLOv8 모델)
model = YOLO("yolov8n-pose.pt", verbose=False)

save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

csv_file = os.path.join(save_dir, "keypoints.csv")

columns = ["Image","label","Nose_x", "Nose_y", "Left Eye_x", "Left Eye_y", "Right Eye_x", "Right_Eye_y", "Left Ear_x", "Left Ear_y", "Right Ear_x", "Right Ear_y", 
           "Left Shoulder_x", "Left Shoulder_y", "Right Shoulder_x", "Right Shoulder_y", "Left Elbow_x", "Left Elbow_y", "Right Elbow_x", "Right Elbow_y", 
           "Left Wrist_x", "Left Wrist_y", "Right Wrist_x", "Right Wrist_y", "Left Hip_x", "Left Hip_y", "Right Hip_x", "Right Hip_y", 
           "Left Knee_x", "Left Knee_y", "Right Knee_x", "Right Knee_y", "Left Ankle_x", "Left Ankle_y", "Right Ankle_x", "Right Ankle_y"]
class_name = "fallperson" # person, walkewithphone, fallperson
# CSV 파일 초기화
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file, index=False)

# 웹캠에서 프레임을 캡처하고 처리하는 루프
while True:
    ret, frame = cap.read()
    if not ret:
        cv2.waitKey()
        break

    # YOLO 모델을 사용하여 포즈 감지 수행
    results = model(frame)
    print(results)
    # 감지된 객체 및 포즈 시각화
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        keypoints = result.keypoints.xy.cpu().numpy()  # Keypoints
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(result.boxes.cls[i])  # 클래스 ID 얻기
            label = model.names[cls_id]  # Object name 얻기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
            person_keypoints = keypoints[i]
            for point in person_keypoints:
                x, y = map(int, point)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
    # 결과 프레임을 디스플레이
    cv2.imshow('YOLOv8 Pose Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # 's' 키를 누르면 이미지 저장 및 키포인트 CSV 파일에 기록
    if key == ord('s'):
        # 이미지 파일 이름 설정 (현재 시간 또는 고유 이름 사용)
        img_name = os.path.join(save_dir, f"capture_{cv2.getTickCount()}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"이미지 저장: {img_name}")

        # 키포인트 데이터 기록
        if keypoints.size > 0:
            keypoints_flat = keypoints.flatten()
            data = [img_name] +[class_name] +keypoints_flat.tolist()
            if len(data) == len(columns):
                df = pd.read_csv(csv_file)
                df.loc[len(df)] = data
                df.to_csv(csv_file, index=False)
                print(f"키포인트 데이터 저장: {csv_file}")
            else:
                print("키포인트 데이터 길이가 일치하지 않습니다.")

# 리소스 해제
cap.release()
cv2.destroyAllWindows()