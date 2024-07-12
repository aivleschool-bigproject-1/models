import cv2
import time
import math as m
import mediapipe as mp


# 거리 계산
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# 각도 계산
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi * theta)
    return degree


"""
경고를 보내는 함수입니다. 나쁜 자세가 감지되었을 때 이 함수를 사용하여 경고를 보냅니다.
원하는 대로 창의적으로 사용자 정의할 수 있습니다.
"""
def sendWarning(x):
    pass

# =============================상수 및 초기화=====================================#
# 프레임 카운터 초기화
good_frames = 0
bad_frames = 0

# 폰트 유형
font = cv2.FONT_HERSHEY_SIMPLEX

# 색상
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Mediapipe 포즈 클래스 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#

if __name__ == "__main__":
    # 웹캠 입력을 위해 파일 이름을 0으로 교체하십시오.
    cap = cv2.VideoCapture(0)

    # 메타 데이터
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    while cap.isOpened():
        # 프레임 캡처
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # fps 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 높이와 너비 가져오기
        h, w = image.shape[:2]

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 처리
        keypoints = pose.process(image)

        # 이미지를 다시 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # lm과 lmPose를 다음 메서드의 대표로 사용
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        if lm:
            # 랜드마크 좌표 얻기
            # 왼쪽 어깨
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # 오른쪽 어깨
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # 코
            nose_x = int(lm.landmark[lmPose.NOSE].x * w)
            nose_y = int(lm.landmark[lmPose.NOSE].y * h)

            # 왼쪽과 오른쪽 어깨 높이 차이 계산
            shoulder_diff = abs(l_shldr_y - r_shldr_y)

            # 코와 어깨의 x 좌표 차이 계산 (목이 앞으로 기울어졌는지 확인)
            forward_head_position = abs(nose_y - (l_shldr_y + r_shldr_y) / 2)
            overlay = image.copy()
            alpha = 0.3
            cv2.ellipse(overlay, (335, 170), (60, 90), 0, 0, 360, (255, 255, 255), -1)
            # 랜드마크 그리기
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (nose_x, nose_y), 7, blue, -1)
            cv2.addWeighted(overlay,alpha ,image, 1-alpha, 0, image)
            # 텍스트 넣기, 자세 및 각도 기울기
            angle_text_string = 'Shoulder diff : ' + str(int(shoulder_diff)) + ' degrees'
            position_text_string = 'Forward Head Position : ' + str(int(forward_head_position)) + ' px'

            # 좋은 자세인지 나쁜 자세인지 결정
            if shoulder_diff < 50 and forward_head_position > 180:
                bad_frames = 0
                good_frames += 1
                
                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                cv2.putText(image, position_text_string, (10, 60), font, 0.9, light_green, 2)
                cv2.putText(image, 'Good Posture', (10, 90), font, 0.9, light_green, 2)

                # 랜드마크 연결
                cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), green, 4)
                cv2.line(image, (nose_x, nose_y), (int((l_shldr_x + r_shldr_x) / 2), int((l_shldr_y + r_shldr_y) / 2)), green, 4)

            else:
                good_frames = 0
                bad_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                cv2.putText(image, position_text_string, (10, 60), font, 0.9, red, 2)
                cv2.putText(image, 'Bad Posture', (10, 90), font, 0.9, red, 2)

                # 랜드마크 연결
                cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), red, 4)
                cv2.line(image, (nose_x, nose_y), (int((l_shldr_x + r_shldr_x) / 2), int((l_shldr_y + r_shldr_y) / 2)), red, 4)

            # 특정 자세를 유지한 시간을 계산
            good_time = (1 / fps) * good_frames
            bad_time =  (1 / fps) * bad_frames

            # 자세 유지 시간
            if good_time > 0:
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

            # 나쁜 자세를 3분(180초) 이상 유지하면 경고를 보냅니다.
            if bad_time > 180:
                sendWarning()

        # 화면에 표시
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
