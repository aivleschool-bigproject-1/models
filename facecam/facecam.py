# main.py
import cv2
import numpy as np
from heart_rate_monitor import HeartRateMonitor
from posture_monitor import PostureMonitor
import time

def main():
    cap = cv2.VideoCapture(0)
    heart_rate_monitor = HeartRateMonitor()
    posture_monitor = PostureMonitor()

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        graph_width = frame_width 
        graph_height = frame_height
        output_width = frame_width + graph_width
        output_height = frame_height
        output_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 0

        # 심박수 모니터링
        heart_rate_monitor.run(frame)

        # 자세 모니터링
        posture_result = posture_monitor.run(frame)

        # 출력을 output_frame에 합치기
        output_frame[:, :frame_width] = posture_result
        bpm_graph = heart_rate_monitor.plot_bpm(graph_width, graph_height)
        output_frame[:, frame_width:frame_width + graph_width] = bpm_graph

        cv2.imshow('Frame', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        print(f"Frame time: {end_time - start_time:.4f} seconds")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
