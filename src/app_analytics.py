import time
import csv
import os
from collections import defaultdict
import cv2
from ultralytics import YOLO

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    model = YOLO("yolov8n.pt")

    # Tracking generator (ByteTrack)
    results_gen = model.track(
        source=0,                  # 0 = webcam; change to a video path if preferred
        tracker="bytetrack.yaml",
        conf=0.5,
        stream=True,
        verbose=False,
        persist=True
    )

    # Analytics state
    start_time = time.time()
    first_seen = {}               # id -> first timestamp
    last_seen = {}                # id -> last timestamp
    total_dwell = defaultdict(float)  # id -> total seconds
    id_class = {}                 # id -> class name
    class_seen_once = set()       # (class_name, id) pairs to count unique entries
    class_unique_counts = defaultdict(int)  # class_name -> unique object count

    fps_last_time = time.time()
    frame_count = 0

    for r in results_gen:
        frame = r.plot()

        # Parse track IDs and classes if available
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().tolist()
            clss = r.boxes.cls.cpu().tolist() if r.boxes.cls is not None else []
            names = r.names

            now = time.time()
            for i, tid in enumerate(ids):
                # Map class
                cls_name = names[int(clss[i])] if i < len(clss) else "unknown"
                id_class[tid] = cls_name

                # Initialize first_seen
                if tid not in first_seen:
                    first_seen[tid] = now
                    last_seen[tid] = now
                    # Unique class count
                    key = (cls_name, tid)
                    if key not in class_seen_once:
                        class_seen_once.add(key)
                        class_unique_counts[cls_name] += 1
                else:
                    # accumulate dwell based on time since last_seen
                    dt = now - last_seen[tid]
                    if dt > 0:
                        total_dwell[tid] += dt
                    last_seen[tid] = now

        # Simple FPS overlay
        frame_count += 1
        now_disp = time.time()
        elapsed = now_disp - fps_last_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_last_time = now_disp
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw a small analytics panel
        y0 = 60
        cv2.putText(frame, "Unique counts:", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y = y0 + 25
        shown = 0
        for cls_name, cnt in sorted(class_unique_counts.items(), key=lambda x: -x[1]):
            if shown >= 6:  # avoid clutter; show top 6
                break
            cv2.putText(frame, f"{cls_name}: {cnt}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 25
            shown += 1

        cv2.imshow("Detection + Tracking + Analytics — press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Prepare CSV log
    ensure_dir("outputs")
    log_path = os.path.join("outputs", "detections_log.csv")
    session_end = time.time()

    # Write per-ID dwell summary
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "class_name", "first_seen_ts", "last_seen_ts", "total_dwell_sec"])
        for tid, cls_name in id_class.items():
            fs = first_seen.get(tid, session_end)
            ls = last_seen.get(tid, session_end)
            dwell = total_dwell.get(tid, max(0.0, ls - fs))  # fallback if not accumulated
            writer.writerow([tid, cls_name, f"{fs:.3f}", f"{ls:.3f}", f"{dwell:.3f}"])

    print(f"[INFO] Saved analytics log to {log_path}")

if __name__ == "__main__":
    main()
