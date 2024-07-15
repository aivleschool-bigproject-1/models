import cv2
import dlib
import numpy as np
import time
from scipy.signal import butter, lfilter

class HeartRateMonitor:
    def __init__(self, buffer_size=150):
        # 얼굴 검출기, 랜드마크 예측기 초기화
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # 심박수 계산 초기 변수 설정
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.times = []
        self.bpm = 0
        self.t0 = time.time()

    # 입력 이미지에서 얼굴 검출, 랜드마크 추출
    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) > 0:
            landmarks = self.predictor(gray, faces[0])
            return [(p.x, p.y) for p in landmarks.parts()]
        return None

    # 랜드마크를 이용해 이마 영역을 추출
    def get_forehead_roi(self, landmarks, image):
        if landmarks:
            x1, y1 = landmarks[21]
            x2, y2 = landmarks[22]
            forehead = image[y1-15:y1+15, x1-15:x2+15]
            return forehead
        return None

    # 이마 영역 녹색 채널 평균값 계산
    def extract_green_channel_mean(self, roi):
        if roi is not None and roi.size > 0:
            green_channel = roi[:, :, 1]
            return np.mean(green_channel)
        return None

    # 프레임 처리, 심박수 계산
    def run(self, frame):
        landmarks = self.get_landmarks(frame)
        roi = self.get_forehead_roi(landmarks, frame)
        green_mean = self.extract_green_channel_mean(roi)

        if green_mean is not None:
            self.data_buffer.append(green_mean)
            self.times.append(time.time() - self.t0)

            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer.pop(0)
                self.times.pop(0)
                self.compute_heart_rate()

    # 밴드패스 필터 적용(심박수 측정 대역대 추출)
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    # 심박수 계산
    def compute_heart_rate(self):
        fs = len(self.data_buffer) / (self.times[-1] - self.times[0])
        filtered = self.butter_bandpass_filter(self.data_buffer, 0.75, 4.0, fs, order=5)
        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), 1.0/fs)
        idx = np.argmax(fft)
        self.bpm = freqs[idx] * 60.0
        print(f"BPM: {self.bpm:.2f}")

def main():
    cap = cv2.VideoCapture(0)
    monitor = HeartRateMonitor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        monitor.run(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
