# 🧠 Realtime Object Analytics (YOLOv8 + Streamlit)

> Interactive web app for **real-time detection and analysis** of common indoor objects.  
> **Detected Classes:** person, chair, cell phone, cup, bottle, couch, dining table  

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10–3.13-blue" />
  <img src="https://img.shields.io/badge/ultralytics-8.3.217-orange" />
  <img src="https://img.shields.io/badge/streamlit-app-red" />
</p>

---

## 📋 Full Setup + Guide (Single Copyable Section)

```bash
# 🚀 QUICKSTART

# 1️⃣ Create and activate environment
python -m venv .venv
.\.venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run src/app_streamlit.py --server.port 8501


# 📁 PROJECT STRUCTURE

models/
 ├── best_refit.pt          # Main fine-tuned YOLOv8 weights
 └── best1.pt               # Earlier optional model

src/
 └── app_streamlit.py       # Main Streamlit interface

test_images/                # Sample test images
videos/                     # Optional demo clips
room_train_as_val.yaml      # Dataset reference file
requirements.txt            # Dependencies list


# 🧩 FEATURES

✅ Real-time object detection using YOLOv8
✅ Streamlit dashboard for image, video, and webcam input
✅ Automatic snapshots and CSV logging
✅ Lightweight analytics per detected class
✅ Adaptive brightness normalization
✅ Sidebar controls for filtering and performance modes


# ⚙️ HOW IT WORKS

1. Input: User uploads image/video or enables webcam.
2. Detection: YOLOv8 model (best_refit.pt) runs inference in real time.
3. Post-Processing: Confidence filtering, normalization, and analytics logging.
4. Analytics: Summaries stored as CSV with per-class counts and timestamps.
5. Display: Streamlit dashboard shows annotated frames and detection stats.


# 📈 MODEL INFORMATION

Model Name      : best_refit.pt
Base Framework  : YOLOv8 (Ultralytics)
Classes         : 7 indoor classes
Dataset         : Custom room-based dataset
Use Case        : Indoor analytics, robotics, HRI, and CV applications


# 🧰 TECH STACK

Language     : Python 3.10 – 3.13
Frameworks   : Streamlit, Ultralytics YOLOv8
Utilities    : OpenCV, Pillow, Pandas, Matplotlib


# 📸 DEMO PREVIEW

# Add your screenshots or demo videos under a docs/ folder, for example:
# (These will automatically render on GitHub)

![App Demo](docs/demo_screenshot.png)
![Sidebar Controls](docs/sidebar_view.png)
![Webcam Mode](docs/webcam_demo.gif)


# 🌟 FUTURE ADD-ONS

• Performance presets (Speed / Balanced / Quality)
• Hybrid snapshots (contact sheet + configurable frames)
• Frame pacing control (e.g., 20 ms delay option)
• Grad-CAM visualization for explainability
• FPS benchmark mode and ONNX/quantized export
• Domain adaptations (medical / robotics / gesture datasets)
• Auto-generated analytics report (PDF + charts)
