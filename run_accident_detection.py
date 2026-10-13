import os

import cv2
import pygame
from ultralytics import YOLO

pygame.mixer.init()

model = YOLO('best.pt')
use_video_file = False

if use_video_file:
    video_path = os.path.join(os.getcwd(), 'testing1.mp4')
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

alarm_sound = pygame.mixer.Sound('alarm.wav')
output_folder = os.path.join(os.getcwd(), 'runs', 'detect', 'outputs')
os.makedirs(output_folder, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_video_path = os.path.join(output_folder, 'output_video.avi')
output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

if not cap.isOpened():
    raise RuntimeError('Could not open camera / video source.')

alarm_playing = False

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)

        # Always draw all bounding boxes + labels on the frame
        annotated_frame = results[0].plot()

        # Check if any box has confidence > 0.5
        detected = False
        boxes = results[0].boxes.cpu().numpy()
        for box in boxes:
            if box.conf[0] > 0.5:
                detected = True
                break

        # Status banner at the top
        banner_color = (0, 0, 200) if detected else (0, 180, 0)
        status_text = '⚠ ACCIDENT DETECTED!' if detected else '✔ Monitoring...'
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), banner_color, -1)
        cv2.putText(annotated_frame, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence of highest-confidence box (if any)
        if len(boxes) > 0:
            max_conf = float(max(b.conf[0] for b in boxes))
            conf_text = f'Confidence: {max_conf:.2f}'
            cv2.putText(annotated_frame, conf_text, (10, annotated_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

        # Play/stop alarm
        if detected and not alarm_playing:
            alarm_sound.play(-1)   # loop until stopped
            alarm_playing = True
        elif not detected and alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

        output_video.write(annotated_frame)
        cv2.imshow('Accident Detection - Live Camera', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    if alarm_playing:
        alarm_sound.stop()
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

print(f'Processed video saved to: {output_video_path}')
