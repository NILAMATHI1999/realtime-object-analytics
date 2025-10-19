# src/extract_frames.py
import cv2
import os
import glob
import math
import sys

def extract_frames(video_path, output_dir, target_fps=2):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[ERROR] Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not math.isfinite(video_fps) or video_fps <= 0:
        print("[WARN] Video FPS unknown or zero. Assuming 30 FPS.")
        video_fps = 30.0

    frame_interval = max(1, int(round(video_fps / target_fps)))
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Detected FPS: {video_fps:.2f} | Target FPS: {target_fps} | Interval: {frame_interval}")

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved} frames to {output_dir}")

def find_default_video():
    patterns = ["videos/*.mp4", "videos/*.mov", "videos/*.MOV", "videos/*.avi", "videos/*.mkv"]
    for p in patterns:
        found = glob.glob(p)
        if found:
            return found[0]
    return None

if __name__ == "__main__":
    # Usage:
    #   python src/extract_frames.py                -> auto-picks first video in videos/
    #   python src/extract_frames.py path\to.mp4   -> uses the provided path
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        video = find_default_video()
        if video is None:
            raise FileNotFoundError("[ERROR] No video found. Put one in the 'videos' folder or pass a path.")

    extract_frames(video, "data/raw_frames", target_fps=2)
