import time
import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    # Use Ultralytics built-in ByteTrack
    # If webcam index 0 fails, try 1 or a video path like "data\\sample.mp4"
    results_gen = model.track(
        source=0,
        tracker="bytetrack.yaml",  # built-in tracker config
        conf=0.5,
        stream=True,               # yields results per frame
        verbose=False,
        persist=True               # keep track IDs
    )

    fps_last_time = time.time()
    frame_count = 0

    for r in results_gen:
        frame_count += 1
        # r contains detections + track IDs; r.plot() draws boxes and IDs
        frame = r.plot()

        # Simple FPS estimate
        now = time.time()
        elapsed = now - fps_last_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Tracking (YOLOv8 + ByteTrack) — press q to quit", frame)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
