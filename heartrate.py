import cv2
import dlib
import numpy as np
import time
from scipy.signal import butter, lfilter
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
        
        # 그래프 설정
        self.bpm_values = deque(maxlen=buffer_size)

    # 입력 이미지에서 얼굴 검출, 랜드마크 추출
    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) > 0:
            landmarks = self.predictor(gray, faces[0])
            return [(p.x, p.y) for p in landmarks.parts()], faces[0]
        return None, None

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
        landmarks, face = self.get_landmarks(frame)
        if face is not None:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
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
        self.bpm_values.append(self.bpm)
        print(f"BPM: {self.bpm:.2f}")

    # 실시간 그래프 업데이트
    def plot_bpm(self, width, height):
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        if len(self.bpm_values) > 1:
            ax.plot(self.bpm_values, color='green')
        ax.set_xlim([0, self.buffer_size])
        ax.set_ylim([0, 200])
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('BPM', color='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        graph = np.asarray(buf)
        graph = cv2.cvtColor(graph, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return graph

def main():
    cap = cv2.VideoCapture(0)
    monitor = HeartRateMonitor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        graph_width = int(frame_width * 2 / 5)
        graph_height = int(frame_height / 3)
        output_width = frame_width + graph_width
        output_height = frame_height
        output_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
        
        monitor.run(frame)

        # 웹캠 화면을 output_frame의 왼쪽 3/5 영역에 넣기
        output_frame[:, :frame_width] = frame

        # BPM 그래프
        bpm_graph = monitor.plot_bpm(graph_width, graph_height)
        output_frame[:graph_height, frame_width:frame_width + graph_width] = bpm_graph

        cv2.imshow('Frame', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
