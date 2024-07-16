# heart_rate_monitor.py
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
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.times = []
        self.bpm = 0
        self.t0 = time.time()
        self.bpm_values = deque(maxlen=buffer_size)

    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) > 0:
            landmarks = self.predictor(gray, faces[0])
            return [(p.x, p.y) for p in landmarks.parts()], faces[0]
        return None, None

    def get_forehead_roi(self, landmarks, image):
        if landmarks:
            x1, y1 = landmarks[21]
            x2, y2 = landmarks[22]
            forehead = image[y1-15:y1+15, x1-15:x2+15]
            return forehead
        return None

    def extract_green_channel_mean(self, roi):
        if roi is not None and roi.size > 0:
            green_channel = roi[:, :, 1]
            return np.mean(green_channel)
        return None

    def run(self, frame):
        landmarks, face = self.get_landmarks(frame)
        if face is not None:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
        roi = self.get_forehead_roi(landmarks, frame)
        green_mean = self.extract_green_channel_mean(roi)

        if green_mean is not None:
            self.data_buffer.append(green_mean)
            self.times.append(time.time() - self.t0)

            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer.pop(0)
                self.times.pop(0)
                self.compute_heart_rate()

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def compute_heart_rate(self):
        fs = len(self.data_buffer) / (self.times[-1] - self.times[0])
        filtered = self.butter_bandpass_filter(self.data_buffer, 0.75, 4.0, fs, order=5)
        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), 1.0/fs)
        idx = np.argmax(fft)
        self.bpm = freqs[idx] * 60.0
        self.bpm_values.append(self.bpm)
        print(f"BPM: {self.bpm:.2f}")

    def plot_bpm(self, width, height):
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')
        
        ax.grid(which='both', color='#1C3554', linestyle='-', linewidth=0.5)
        
        if len(self.bpm_values) > 1:
            ax.plot(self.bpm_values, color='#00FFFF', linewidth=2)
        ax.set_xlim([0, self.buffer_size])
        ax.set_ylim([0, 200])
        ax.set_xlabel('Time', color='#00FFFF')
        ax.set_ylabel('BPM', color='#00FFFF')
        ax.xaxis.label.set_color('#00FFFF')
        ax.yaxis.label.set_color('#00FFFF')
        ax.tick_params(axis='x', colors='#00FFFF')
        ax.tick_params(axis='y', colors='#00FFFF')
        for spine in ax.spines.values():
            spine.set_edgecolor('#00FFFF')

        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        graph = np.asarray(buf)
        graph = cv2.cvtColor(graph, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return graph
