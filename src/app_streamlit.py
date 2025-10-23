# streamlit_app.py
# Robust YOLO inference app for videos/webcam with adaptive fallbacks

import os, time, tempfile, subprocess, shutil, warnings
from typing import Dict, Tuple, Union, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
try:
    cv2.setNumThreads(0)
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Realtime Object Analytics (Robust)", layout="wide")

# ========= Fixed classes (match your dataset order) =========
SHOW_CLASSES: List[str] = [
    "person", "chair", "cell phone", "cup", "bottle", "couch", "dining table"
]

PALETTE: List[Tuple[int, int, int]] = [
    (0, 165, 255),   # person  -> orange
    (0, 255, 0),     # chair   -> green
    (255, 0, 0),     # cell phone -> blue
    (255, 0, 255),   # cup     -> magenta
    (0, 255, 255),   # bottle  -> yellow
    (255, 255, 0),   # couch   -> cyan
    (180, 105, 255), # dining table -> pink-ish
]
assert len(PALETTE) == len(SHOW_CLASSES), "PALETTE must match SHOW_CLASSES length."

CLASS_COLOR: Dict[str, Tuple[int, int, int]] = {name: PALETTE[i] for i, name in enumerate(SHOW_CLASSES)}

def get_color(name: str) -> Tuple[int, int, int]:
    if name in CLASS_COLOR:
        return CLASS_COLOR[name]
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return tuple(int(c) for c in rng.integers(0, 255, size=3))

# ========= Model helpers =========
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    return YOLO(weights_path)

def get_name_maps(model: YOLO):
    names = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", {})
    if isinstance(names, list):
        id2name = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        id2name = {int(k): v for k, v in names.items()}
    else:
        id2name = {}
    name2id = {v: k for k, v in id2name.items()}
    return id2name, name2id

# ========= Drawing =========
def draw_box_label(img, x1, y1, x2, y2, label, color):
    h, w = img.shape[:2]
    scale = max(0.6, min(1.6, w / 640.0))
    th_box = max(2, int(2.2 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.7, 0.9 * scale)
    th_txt = max(2, int(2 * scale))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, th_box)

    (tw, th), _ = cv2.getTextSize(label, font, fs, th_txt)
    pad = max(4, int(5 * scale))
    bx1, by1 = x1, max(0, y1 - th - 2 * pad)
    bx2, by2 = min(w, x1 + tw + 2 * pad), y1

    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    tx, ty = bx1 + pad, by2 - pad
    cv2.putText(img, label, (tx, ty), font, fs, (0, 0, 0), max(1, th_txt + 2), cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), font, fs, (245, 245, 245), th_txt, cv2.LINE_AA)

# ========= Light normalization (helps dim/backlit) =========
def normalize_light(frame: np.ndarray) -> np.ndarray:
    # YCrCb CLAHE on Y channel
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y2 = clahe.apply(y)
    ycc2 = cv2.merge([y2, cr, cb])
    out = cv2.cvtColor(ycc2, cv2.COLOR_YCrCb2BGR)
    return out

# ========= Core inference on one frame with fallbacks =========
def infer_with_fallbacks(model: YOLO, frame: np.ndarray, *,
                         conf: float, iou: float, imgsz: int,
                         max_det: int, classes: List[int],
                         robust: bool):
    # pass 1: user settings
    res = model.predict(
        frame, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
        agnostic_nms=True, classes=classes or None,
        augment=False, verbose=False
    )[0]
    if len(res.boxes) > 0 or not robust:
        return res

    # pass 2: lower conf + TTA
    res = model.predict(
        frame, conf=max(0.05, conf * 0.7), iou=min(0.65, iou + 0.05),
        imgsz=max(imgsz, 640), max_det=max_det,
        agnostic_nms=True, classes=classes or None,
        augment=True, verbose=False
    )[0]
    if len(res.boxes) > 0:
        return res

    # pass 3: light normalization + TTA + low conf
    norm = normalize_light(frame)
    res = model.predict(
        norm, conf=0.05, iou=min(0.7, iou + 0.1),
        imgsz=640, max_det=max_det,
        agnostic_nms=True, classes=classes or None,
        augment=True, verbose=False
    )[0]
    return res

# ========= Video/stream loop =========
def run_stream(
    source: Union[int, str],
    model: YOLO,
    conf: float,
    iou: float,
    allowed_class_names: List[str],
    id2name: Dict[int, str],
    name2id: Dict[str, int],
    imgsz: int = 384,
    vid_stride: int = 6,
    max_frames: int = 140,
    max_runtime_sec: int = 120,
    max_det: int = 60,
    robust: bool = True,
    save_full_video: bool = True,
    make_preview: bool = True,
    preview_width: int = 640,
    preview_fps: int = 10,
    full_video_fps: float = 12.0,
    enable_throttle: bool = True,
    ui_sleep_ms: int = 60,
):
    allowed_ids = [name2id[n] for n in allowed_class_names if n in name2id]
    allowed_set = set(allowed_class_names)

    t0 = time.time()
    os.makedirs("outputs", exist_ok=True)
    paths = {
        "csv": "outputs/detections_log.csv",
        "video": "outputs/annotated_full.mp4",
        "video_preview": "outputs/annotated_preview.mp4",
        "snapshot_first": "outputs/snapshot_first.png",
        "snapshot_best": "outputs/snapshot_best.png",
        "snapshot_contact": "outputs/snapshot_contact.png",
    }

    progress = st.progress(0)
    status_text = st.empty()

    cap = cv2.VideoCapture(0 if isinstance(source, int) else str(source))
    if not cap.isOpened():
        st.error("Could not open source.")
        return pd.DataFrame([], columns=["class_name","t_sec"]), {}, paths, 0.0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    est = min(max_frames, (total_frames + vid_stride - 1)//vid_stride) if total_frames > 0 else max_frames

    records = []
    class_first = {}
    best_frame, best_count = None, 0
    first_saved = False

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fidx, processed = 0, 0

    while True:
        if (time.time() - t0) > max_runtime_sec:
            status_text.warning("⏱️ Stopped: runtime cap.")
            break
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx % vid_stride) != 0:
            fidx += 1
            continue

        result = infer_with_fallbacks(
            model, frame, conf=conf, iou=iou, imgsz=imgsz,
            max_det=max_det, classes=allowed_ids, robust=robust
        )

        ann = frame.copy()
        found = set()

        if getattr(result, "boxes", None) is not None and getattr(result.boxes, "cls", None) is not None:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                cls_name = id2name.get(cls_id, "unknown")
                if cls_name not in allowed_set:
                    continue
                found.add(cls_name)
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                cscore = float(result.boxes.conf[i]) if getattr(result.boxes, "conf", None) is not None else 0.0
                draw_box_label(ann, x1, y1, x2, y2, f"{cls_name} {cscore:.2f}", get_color(cls_name))

        t_sec = int(round(fidx / max(1.0, fps_in)))
        for c in found:
            records.append({"class_name": c, "t_sec": t_sec})
            if c not in class_first:
                class_first[c] = ann.copy()

        if len(found) > best_count:
            best_count = len(found)
            best_frame = ann.copy()

        if not first_saved:
            cv2.imwrite(paths["snapshot_first"], ann)
            first_saved = True

        if save_full_video:
            h, w = ann.shape[:2]
            out_fps = min(full_video_fps, max(5.0, fps_in / max(1, vid_stride)))
            if writer is None:
                writer = cv2.VideoWriter(paths["video"], fourcc, out_fps, (w, h))
            writer.write(ann)

        processed += 1
        fidx += 1

        progress.progress(int(min(100, (processed / max(1, est)) * 100)))
        status_text.write(f"Processed {processed} / ~{est}")

        if processed >= max_frames:
            status_text.warning("🛑 Stopped: frame cap.")
            break
        if enable_throttle and ui_sleep_ms > 0:
            time.sleep(ui_sleep_ms / 1000.0)

    cap.release()
    if writer is not None:
        writer.release()

    if best_frame is not None:
        cv2.imwrite(paths["snapshot_best"], best_frame)

    # Contact sheet
    if class_first:
        imgs = []
        for _, fr in class_first.items():
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(rgb))
        if imgs:
            w, h = imgs[0].size
            cols = 3
            rows = (len(imgs) + cols - 1) // cols
            sheet = Image.new("RGB", (cols*w, rows*h), (255, 255, 255))
            for i, im in enumerate(imgs):
                sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
            sheet.save(paths["snapshot_contact"])

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=["class_name","t_sec"])
    counts = df["class_name"].value_counts().to_dict() if not df.empty else {}
    df.to_csv(paths["csv"], index=False)

    # Optional H.264 preview via ffmpeg if found
    if os.path.exists(paths["video"]):
        ffmpeg = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
        if os.path.exists(ffmpeg):
            try:
                cmd = [
                    ffmpeg, "-y", "-i", paths["video"], "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline",
                    "-level", "3.1", "-movflags", "+faststart", "-r", str(preview_fps),
                    "-vf", f"scale='min({preview_width},iw)':-2", paths["video_preview"]
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=(0x08000000 if os.name == "nt" else 0))
            except Exception:
                pass

    return df, counts, paths, round(time.time() - t0, 2), []

# ========= UI =========
def main():
    st.title("Realtime Object Analytics (Robust Mode)")
    st.caption("TTA + adaptive thresholds + light normalization for tough conditions")

    st.sidebar.title("⚙️ Controls")
    default_weights = "models/best1.pt"  # change if needed
    weights_path = st.sidebar.text_input("Model weights (.pt)", value=default_weights)

    # Load model + names
    try:
        model = load_model(weights_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    id2name, name2id = get_name_maps(model)

    st.sidebar.markdown("**Classes (fixed order):**")
    st.sidebar.write(", ".join(SHOW_CLASSES))

    # Inference settings
    robust = st.sidebar.toggle("Robust mode (recommended)", True)
    conf = st.sidebar.slider("Base confidence", 0.05, 0.90, 0.35, 0.05)
    iou = st.sidebar.slider("NMS IoU", 0.30, 0.80, 0.60, 0.05)
    imgsz = st.sidebar.select_slider("Image size", options=[320, 384, 448, 512, 640], value=384)
    vid_stride = st.sidebar.select_slider("Frame stride", options=[1, 2, 4, 6, 8], value=4)
    max_frames = st.sidebar.select_slider("Max frames", options=[80, 100, 140, 180, 240], value=140)
    ui_sleep_ms = st.sidebar.slider("Throttle per frame (ms)", 0, 120, 60, 5)

    use_webcam = st.sidebar.toggle("Use webcam", False)
    uploaded, source_val, temp_file = None, None, None
    if not use_webcam:
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "mpeg4"])
    run_btn = st.sidebar.button("▶️ Run")

    if run_btn:
        if use_webcam:
            source_val = 0
        elif uploaded:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded.read()); temp_file.flush()
            source_val = temp_file.name
        else:
            st.warning("Upload a video or enable webcam.")
            st.stop()

        with st.spinner("Processing…"):
            df, counts, paths, runtime_sec, _ = run_stream(
                source=source_val,
                model=model,
                conf=float(conf),
                iou=float(iou),
                allowed_class_names=SHOW_CLASSES,
                id2name=id2name, name2id=name2id,
                imgsz=int(imgsz), vid_stride=int(vid_stride),
                max_frames=int(max_frames), max_runtime_sec=180,
                max_det=100, robust=robust,
                save_full_video=True, make_preview=True,
                preview_width=640, preview_fps=10,
                full_video_fps=12.0, enable_throttle=True,
                ui_sleep_ms=int(ui_sleep_ms),
            )

        st.session_state.update(df=df, counts=counts, paths=paths, runtime_sec=runtime_sec)
        st.success("Done ✅")
        st.info(f"{len(df)} records • {len(counts)} classes • {runtime_sec}s runtime")

    # Snapshots
    if "paths" in st.session_state:
        paths = st.session_state["paths"]
        st.subheader("📸 Snapshots")
        c1, c2, c3 = st.columns(3)
        with c1:
            p = paths.get("snapshot_first", "")
            if os.path.exists(p): st.image(p, caption="First annotated")
        with c2:
            p = paths.get("snapshot_best", "")
            if os.path.exists(p): st.image(p, caption="Best unique classes")
        with c3:
            p = paths.get("snapshot_contact", "")
            if os.path.exists(p): st.image(p, caption="Contact sheet")

    # Analytics & downloads
    if "df" in st.session_state and "counts" in st.session_state:
        df, counts, paths = st.session_state["df"], st.session_state["counts"], st.session_state["paths"]
        st.subheader("📊 Analytics")
        if counts:
            chart_df = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())})
            st.bar_chart(chart_df.set_index("class"))
            fig, ax = plt.subplots()
            ax.pie(chart_df["count"], labels=chart_df["class"], autopct='%1.1f%%')
            st.pyplot(fig)
        st.dataframe(df, use_container_width=True)

        st.subheader("⬇️ Downloads")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               "detections_log.csv", "text/csv")
        with c2:
            vidp = paths.get("video_preview","")
            if os.path.exists(vidp):
                st.video(vidp)
            fullp = paths.get("video","")
            if os.path.exists(fullp):
                with open(fullp, "rb") as f:
                    st.download_button("Download Full Annotated Video", f.read(),
                                       "annotated_full.mp4", "video/mp4")

if __name__ == "__main__":
    main()
