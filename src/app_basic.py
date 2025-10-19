import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # tiny & fast

    cap = cv2.VideoCapture(0)    # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Webcam not found. Try VideoCapture(1) or use a video file path.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Real-Time Detection (YOLOv8n) — press q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
